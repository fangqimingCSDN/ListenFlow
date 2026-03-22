# 实时语音转写系统 — 架构设计文档 v3.0

**版本**: v3.0  |  **日期**: 2026-03-22  |  **作者**: 架构组

> 本文档面向技术评审与架构汇报，涵盖系统定位、整体架构、数据流、
> 并发模型、存储设计、模型选型决策及部署指南。

---

## 目录

1. [系统定位与核心目标](#一系统定位与核心目标)
2. [整体架构图](#二整体架构图)
3. [核心数据流](#三核心数据流)
4. [异步并发模型](#四异步并发模型)
5. [PostgreSQL 表设计](#五postgresql-表设计)
6. [存储层设计](#六存储层设计)
7. [模型选型决策](#七模型选型决策)
8. [项目结构说明](#八项目结构说明)
9. [关键配置速查](#九关键配置速查)
10. [快速启动指南](#十快速启动指南)

---

## 一、系统定位与核心目标

### 1.1 产品定位

> 面向**多人会议场景**的实时语音转写系统，支持说话人自动分离、实时推送、
> 离线存档与语义检索，全程本地部署，数据不出内网。

### 1.2 核心能力矩阵

| 🎯 能力 | ⚙️ 实现方式 | 📊 指标 |
|--------|------------|--------|
| 实时转写 | WebSocket 流式推送，VAD 断句触发 | 停顿后 1~3s 出字 |
| 说话人分离 | ERes2Net 声纹 + 在线余弦聚类 | 自动标注 speaker_0/1/2… |
| 姓名绑定 | REST API 编辑，历史片段同步更新 | 毫秒级生效 |
| 多路并发 | SessionManager 会话隔离 | 每会话独立 VAD + 聚类状态 |
| 离线存档 | MinIO 对象存储 | 原始音频 WAV + 结构化 JSON |
| 语义检索 | Milvus 向量库（可选开启） | 跨会话关键词/语义搜索 |

### 1.3 非功能性指标

| 指标 | 目标值 |
|------|--------|
| 端到端延迟（说话停止→文字出现） | < 3s（GPU），< 8s（CPU）|
| 并发会话数 | ≥ 10 路（单卡 RTX 3090）|
| 音频存储格式 | 16kHz / 16bit / Mono WAV |
| 数据安全 | 全程内网，无第三方 API 依赖 |
| 模型热切换 | 修改 `.env` 重启即可，无需改代码 |

---

## 二、整体架构图

### 2.1 系统分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      客户端（浏览器）                           │
│                                                                 │
│  🎤 麦克风                                                      │
│     └─ AudioContext ScriptProcessor                            │
│         └─ 16kHz PCM → Float32→Int16 → base64                 │
│             └─ WebSocket ──────────────────────────────────►   │
│  ◄─── {speaker, text, start, end} 实时推送                     │
│  REST: PUT /speakers  GET /download  GET /transcript           │
└──────────────────────────┬──────────────────────────────────────┘
                           │ WebSocket + HTTP
┌──────────────────────────▼──────────────────────────────────────┐
│              FastAPI 后端（单进程 asyncio 事件循环）             │
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │  WebSocket 处理器   │    │  REST API (sessions.py)     │    │
│  │  ws_handler.py      │    │  编辑说话人/下载/查询/列表  │    │
│  └──────────┬──────────┘    └─────────────────────────────┘    │
│             │                                                   │
│  ┌──────────▼──────────────────────────────────────────────┐   │
│  │                    Services 层                          │   │
│  │                                                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │   │
│  │  │ VADService  │  │ ASRService  │  │SpeakerService │   │   │
│  │  │ Silero VAD  │  │ FunASR(默认)│  │ ERes2Net 声纹 │   │   │
│  │  │ 每会话独立  │  │ /Whisper   │  │ + 在线聚类    │   │   │
│  │  │             │  │ /Qwen3-ASR │  │ 余弦相似匹配  │   │   │
│  │  │             │  │ Semaphore  │  │               │   │   │
│  │  └─────────────┘  └─────────────┘  └───────────────┘   │   │
│  │                                                         │   │
│  │  ┌─────────────────────────┐  ┌───────────────────┐    │   │
│  │  │    StorageService       │  │  MilvusService    │    │   │
│  │  │ MinIO 音频+文本         │  │  语义向量+检索    │    │   │
│  │  │ 预签名URL直接下载       │  │  (可选开启)       │    │   │
│  │  └─────────────────────────┘  └───────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  SessionManager 全局单例                                 │  │
│  │  Dict[session_id → SpeechSession]                        │  │
│  │  每会话：VAD实例 + 聚类状态 + 音频缓冲 + 片段列表        │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────┬───────────────────┬────────────────────┬────────────────┘
       │                   │                    │
┌──────▼──────┐   ┌────────▼───────┐   ┌───────▼───────┐
│ PostgreSQL  │   │     MinIO      │   │    Milvus     │
│ sessions    │   │ raw_audio.wav  │   │  384维向量    │
│ speakers    │   │ transcript.txt │   │  HNSW 索引    │
│ segments    │   │ transcript.json│   │  (可选)       │
└─────────────┘   └────────────────┘   └───────────────┘
```

### 2.2 技术栈总览

| 层次 | 技术选型 | 版本 | 用途 |
|------|---------|------|------|
| **接入层** | FastAPI + uvicorn | 0.111 | HTTP/WebSocket 服务 |
| **语音活动检测** | Silero VAD | 5.1 | 实时端点检测 |
| **语音识别** | FunASR paraformer-zh | 1.1.6 | 中文转写（默认）|
| **语音识别备选** | faster-whisper large-v3 | 1.0.3 | 多语言转写 |
| **语音识别增强** | Qwen3-ASR-1.7B | latest | 最高中文精度 |
| **说话人识别** | ERes2Net (modelscope) | latest | 192维声纹提取 |
| **文本向量化** | MiniLM-L12-v2 multilingual | 3.0.1 | 384维语义向量 |
| **关系数据库** | PostgreSQL + asyncpg | 16 | 元数据持久化 |
| **对象存储** | MinIO | latest | 音频/文本归档 |
| **向量数据库** | Milvus | 2.4 | 语义检索（可选）|

---

## 三、核心数据流：一句话如何被识别

### 3.1 完整处理链路（9 步）

```
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1  用户开口说话                                           │
│  浏览器 AudioContext ScriptProcessor                            │
│  每 256ms 采集 4096 样本（16kHz）                               │
│  Float32 → Int16 → base64 → WebSocket 发送                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STEP 2  ws_handler.py 收到 {"type":"audio","data":"<base64>"}  │
│  base64 解码 → PCM bytes（8192字节/包，256ms音频）              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STEP 3  VADService.process_chunk(bytes)  每会话独立实例        │
│                                                                 │
│  bytes → float32 → 追加 buffer                                 │
│  while len(buffer) >= 512:                                      │
│      取 512 样本(32ms) → Silero 神经网络 → 概率 p              │
│                                                                 │
│      p > 0.5 且持续 → 触发 start，记录开始时间                 │
│      p < 0.5 且静音 > 800ms → 触发 end                         │
│      或单段时长 > 30s → 强制截断                               │
│                                                                 │
│  返回 [(start_sec, end_sec, audio_float32_array)]              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STEP 4  asyncio.create_task(_process_segment(...))             │
│  ⚡ 立即返回，不阻塞 WebSocket 接收循环                         │
│  事件循环继续接收下一个音频包                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  STEP 5  asyncio.gather 并行执行（两个线程池任务同时跑）        │
│                                                                 │
│  ┌─────────────────────────┐  ┌──────────────────────────────┐ │
│  │ ASRService.transcribe   │  │ SpeakerService.extract       │ │
│  │ asyncio.to_thread       │  │ asyncio.to_thread            │ │
│  │ FunASR paraformer-zh    │  │ ERes2Net 声纹提取            │ │
│  │ 耗时 2~5s (GPU)         │  │ 耗时 0.3~0.5s               │ │
│  │ 返回: text, confidence  │  │ 返回: 192维 np.ndarray       │ │
│  └──────────┬──────────────┘  └──────────────┬───────────────┘ │
│             └──────────────┬─────────────────┘                 │
│                            │ 两者均完成后继续                  │
└────────────────────────────┼────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 6  OnlineSpeakerCluster.update(embedding)                 │
│                                                                 │
│  新声纹 → 与历史声纹计算余弦相似度                             │
│  相似度 > 0.55 → 归入已有 speaker_N（同一人）                  │
│  相似度 < 0.55 → 新建 speaker_N+1（新说话人）                  │
│  查 speaker_names → 返回 ("speaker_0", "张某某")               │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 7  SpeechSession.add_segment(text, speaker, start, end)   │
│  写入内存片段列表，分配 seq 序号                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 8  fire-and-forget 异步写存储（不等完成直接进下一步）     │
│  create_task(_write_segment_to_db)  → PostgreSQL               │
│  create_task(milvus.insert_segment) → Milvus（如已开启）       │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│  STEP 9  WebSocket.send_json 立即推送给前端                     │
│  {                                                              │
│    "type": "transcript",                                       │
│    "seq": 3,                                                    │
│    "text": "这个方案的价格怎么样",                              │
│    "speaker_label": "speaker_0",                               │
│    "speaker_display": "张某某",                                 │
│    "start": 12.34,  "end": 14.89                               │
│  }                                                              │
│  前端渲染：[张某某] 12:34  这个方案的价格怎么样                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 会话生命周期状态机

```
客户端发送 {"type":"start"}
         │
         ▼
   ┌─────────────┐
   │  recording  │◄──────────────────┐
   └──────┬──────┘                   │
          │ 发送 pause               │ 发送 resume
          ▼                         │
   ┌─────────────┐                   │
   │   paused    │───────────────────┘
   └──────┬──────┘
          │ 发送 stop 或 WebSocket 断开
          ▼
   ┌─────────────┐    后台异步执行
   │ finalizing  │──────────────────────►
   └──────┬──────┘    ① flush_vad() 处理残留片段
          │           ② PCM → WAV → 上传 MinIO
          │           ③ 生成 TXT/JSON → 上传 MinIO
          │           ④ PG: sessions.status = completed
          ▼
   ┌─────────────┐
   │  completed  │
   └─────────────┘
   推送 session_completed（含下载链接）
   ⚡ stop 响应不等待上传完成，延迟 < 1ms
```

---

## 四、异步并发模型详解

### 4.1 为何 ASR 需要 2~5 秒却不阻塞接收新音频？

```
asynio 事件循环（单线程）时间轴

t=0ms    收到音频包#1 → VAD → 检测到语音段A结束
         create_task(处理段A)  ← 登记任务，耗时 < 0.1ms，立即返回
t=1ms    await receive_text()  ← 协程挂起，等待下一包

t=256ms  收到音频包#2 → VAD → 段B 尚未结束
t=512ms  收到音频包#3 → VAD → 检测到段B结束
         create_task(处理段B)  ← 立即返回
t=513ms  继续等待下一个音频包

         ┌────────────────────────────────────────────┐
         │  线程池并行（不占用事件循环）               │
         │  段A: t=1ms   → ASR(2s)‖声纹(0.5s)        │
         │       → t=2001ms 完成                      │
         │  段B: t=513ms → ASR(2s)‖声纹(0.5s)        │
         │       → t=2513ms 完成                      │
         └────────────────────────────────────────────┘

t=2001ms 段A完成 → 事件循环调度 → ws.send_json(transcript_A)
t=2513ms 段B完成 → 事件循环调度 → ws.send_json(transcript_B)

✅ 结论：t=1ms ~ t=2001ms 期间，事件循环持续接收新音频包
         ASR 在线程池运行，完全不占用事件循环线程
```

### 4.2 Semaphore 防止 GPU 显存溢出

```
设 ASR_MAX_CONCURRENT = 2（默认）

同时有 5 个语音段等待转写：

段A → acquire(count=1) → 进线程池 → GPU 推理中...
段B → acquire(count=2) → 进线程池 → GPU 推理中...
段C → acquire → ⏸ 等待（已达上限 2）
段D → acquire → ⏸ 等待
段E → acquire → ⏸ 等待

段A 完成 → release → 段C 进入线程池
段B 完成 → release → 段D 进入线程池

✅ 效果：GPU 同时只跑 2 个 ASR 推理，防止显存 OOM
        等待中的段 C/D/E 以协程挂起形式等待，不阻塞事件循环

⚙️ 调整建议：
   GPU 显存 >= 16GB → ASR_MAX_CONCURRENT=3
   纯 CPU 推理     → ASR_MAX_CONCURRENT=4
```

### 4.3 fire-and-forget 模式（实时性优先）

```
_process_segment 内部执行顺序：

  1. await ASR + 声纹（并行，必须等）
  2. 聚类匹配 speaker（内存操作，< 1ms）
  3. ✅ await ws.send_json(transcript)   ← 先推前端，不等存储
  4. create_task(_write_segment_to_db)   ← PostgreSQL 异步写
  5. create_task(milvus.insert_segment)  ← Milvus 异步写

优先级：实时体验 > 存储延迟
容错性：PG/Milvus 写入失败不影响客户端接收结果
```

### 4.4 并发模型总览

| 组件 | 执行方式 | 是否阻塞事件循环 | 并发控制 |
|------|---------|----------------|----------|
| WebSocket 接收 | asyncio 原生协程 | 否 | 无限制 |
| VAD 处理 | 同步（< 1ms/帧）| 否（极短）| 无需限制 |
| ASR 转写 | asyncio.to_thread | 否 | Semaphore(2) |
| 声纹提取 | asyncio.to_thread | 否 | 同上（gather）|
| 声纹聚类 | 同步（< 1ms）| 否（极短）| 无需限制 |
| PostgreSQL 写 | asyncio + asyncpg | 否 | 连接池 |
| Milvus 写 | asyncio.to_thread | 否 | fire-and-forget |
| MinIO 上传 | asyncio.to_thread | 否 | 会话结束触发 |

---

## 五、PostgreSQL 表设计

### 5.1 三表 ER 关系图

```
┌──────────────────────────────────┐
│            sessions              │
│  id            UUID  PK          │
│  title         VARCHAR           │
│  status        VARCHAR           │  recording / completed
│  language      VARCHAR           │
│  audio_object_key   VARCHAR      │  MinIO 路径
│  transcript_object_key VARCHAR   │  MinIO 路径
│  audio_duration_sec FLOAT        │  冗余字段，避免 SUM
│  speaker_count INT               │  冗余字段，避免 COUNT
│  created_at / completed_at       │
└──────────────┬───────────────────┘
               │ session_id（逻辑关联，无外键）
┌──────────────▼───────────────────┐
│            speakers              │
│  id            UUID  PK          │
│  session_id    UUID  (索引)       │
│  speaker_label VARCHAR           │  speaker_0, speaker_1…
│  display_name  VARCHAR (可空)    │  用户编辑：张某某
│  embedding     JSON              │  声纹备份，主存 Milvus
│  UNIQUE (session_id, label)      │
└──────────────┬───────────────────┘
               │ speaker_id（逻辑关联，无外键）
┌──────────────▼───────────────────┐
│        transcript_segments       │
│  id            UUID  PK          │
│  session_id    UUID  (索引)       │
│  speaker_id    UUID  (索引,可空)  │
│  start_time    FLOAT             │  秒，精度 3 位
│  end_time      FLOAT             │
│  text          TEXT              │  转写文本
│  sequence_no   INT               │  保证排序正确
│  confidence    FLOAT             │  置信度 0~1
│  words         JSON              │  词级时间戳（Whisper）
│  is_final      BOOL              │  区分中间结果和最终结果
└──────────────────────────────────┘
```

### 5.2 为何不使用数据库外键？

| 维度 | 有外键 | 无外键（本项目选择）|
|------|--------|--------------------|
| 写入性能 | 每次 INSERT 需锁定父表行验证 | 直接写入无锁 |
| 高并发 | 行锁竞争严重 | 线性扩展 |
| 分库分表 | 跨库外键不可维护 | 无障碍 |
| 一致性保证 | 数据库强制 | 业务代码逻辑保证 |
| 删除/归档 | 需级联处理 | 灵活独立操作 |

### 5.3 索引策略

```sql
-- speakers 表
CREATE UNIQUE INDEX ON speakers (session_id, speaker_label);

-- transcript_segments 表
CREATE INDEX ON transcript_segments (session_id, sequence_no);  -- 顺序读全部转写
CREATE INDEX ON transcript_segments (session_id, start_time);   -- 播放定位时间区间
CREATE INDEX ON transcript_segments (speaker_id);               -- 按说话人统计
```

---

## 六、存储层设计

### 6.1 三层存储架构

| 🗄️ 层次 | 系统 | 存储内容 | 访问方式 |
|---------|------|---------|----------|
| **元数据层** | PostgreSQL | 会话状态 / 说话人 / 片段文本+时间戳 | SQLAlchemy asyncpg |
| **文件层** | MinIO | 原始录音 WAV + 转写 TXT/JSON | 预签名 URL 直接下载 |
| **向量层** | Milvus（可选）| 文本语义向量 384维 | REST 语义搜索 |

### 6.2 MinIO 存储结构

```
Bucket: speech-audio
  └── {session_id}/
        └── raw_audio.wav          ← 完整原始录音（含静音，会话结束后上传）

Bucket: speech-text
  └── {session_id}/
        ├── transcript.txt         ← 纯文本，按说话人分段
        └── transcript.json        ← 结构化 JSON（含 speaker/时间戳/置信度）
```

> ⚠️ 注意：MinIO 存储的是**完整原始录音**（含静音段），非 VAD 切割后的纯语音。
> 时间轴与 `transcript_segments.start_time` 完全对齐，前端可按时间戳精确定位播放。

### 6.3 Milvus 集合设计

```
集合名：speech_transcripts

字段：
  id            INT64     PK auto_id
  session_id    VARCHAR   按会话过滤
  speaker_label VARCHAR   按说话人过滤
  text          VARCHAR   原始文本
  embedding     FLOAT_VECTOR(384)  MiniLM-L12-v2 语义向量

索引：HNSW  metric=COSINE  M=16  ef_construction=200
分区：按 session_id 分区，单会话查询不扫全表
```

### 6.4 数据流向总结

```
原始音频（每帧）
    │
    ├──► _raw_audio_chunks（内存累积）
    │        └── 会话结束 → get_raw_audio_wav() → MinIO
    │
    └──► VAD 处理
             └── 检测到语音段
                     ├──► ASR → 文本 → PostgreSQL + Milvus
                     └──► 声纹 → 聚类 → PostgreSQL (speakers)
```

---

## 七、模型选型决策

> **选型方法论**：每个组件从五个维度评估：
> 📊 准确率 / ⚡ 延迟 / 💾 资源占用 / 🔧 部署难度 / 🌊 流式兼容性

---

### 7.1 语音识别（ASR）选型

#### 7.1.1 全市场方案横向对比

| 方案 | 开发方 | 架构类型 | 中文 CER | RTF(GPU) | 显存 | 多语言 | 词级时间戳 | 国内部署 |
|------|--------|---------|---------|---------|------|-------|----------|----------|
| ✅ **paraformer-zh** | 达摩院 | 非自回归 CIF | ~3% | 0.05~0.1 | 2 GB | 中文为主 | ❌ | ✅ 免 Token |
| ⭐ **Qwen3-ASR-1.7B** | 阿里云 | LLM Decoder-Only | ~2% | 0.15~0.3 | 4 GB | 50+ 种 | ✅ | ✅ 免 Token |
| SenseVoice-S | 达摩院 | 非自回归 | ~2.5% | 0.03~0.08 | 1.5 GB | 50+ 种 | ✅ | ✅ 免 Token |
| faster-whisper v3 | OpenAI | Encoder-Decoder | ~5% | 0.2~0.4 | 6 GB | 99 种 | ✅ | ✅ 免 Token |
| Whisper medium | OpenAI | Encoder-Decoder | ~7% | 0.1~0.2 | 2.5 GB | 99 种 | ✅ | ✅ 免 Token |
| wav2vec2 | Meta | CTC | ~6% | 快 | 1 GB | 极少 | ❌ | ⚠️ 需申请 |

#### 7.1.2 三大架构本质差异

```
① 非自回归（paraformer）速度最快
   音频 → 一次前向传播 → [你][好][世][界] 同时输出
   RTF ≈ 0.05，速度是 Whisper 的 5~10x

② 自回归 Encoder-Decoder（Whisper）均衡
   音频 → Encoder → Decoder 逐 token 生成
   生成 4 字 = 4 次前向传播  RTF ≈ 0.3
   优势：词级时间戳，99 种语言

③ LLM Decoder-Only（Qwen3-ASR）精度最高
   音频 → 动态编码器 → LLM 自回归生成
   支持 system prompt，无音频长度限制
   中文 CER 最低（~2%），专业术语识别强
```

#### 7.1.3 选型决策

| 后端 | 场景 | 切换方式 |
|------|------|----------|
| ✅ **funasr**（默认）| 中文会议，速度优先 | ASR_BACKEND=funasr |
| faster-whisper | 多语言/词级时间戳 | ASR_BACKEND=whisper |
| Qwen3-ASR | 中文最高精度/专业术语 | ASR_BACKEND=qwen3 |

**未采用**：SenseVoice（与 paraformer 重叠）/ wav2vec2（中文弱无标点）/ Conformer（精度低于离线模式）


### 7.2 语音活动检测（VAD）选型

#### 7.2.1 全市场方案横向对比

| 方案 | 架构 | 检测窗口 | 准确率 | CPU 开销 | 流式 | 部署 |
|------|------|---------|--------|---------|------|------|
| ✅ **Silero VAD** | 小型 LSTM | 32ms | 高 | ~1ms/帧 | ✅ | pip 直装 |
| WebRTC VAD | 传统信号处理 | 10ms | 中 | 极低 | ✅ | C 库绑定 |
| pyannote VAD | Transformer | 帧级 | 极高 | 需 GPU | ❌ | 需 HF Token |
| FSMN-VAD | FSMN 神经网络 | 10ms | 高 | 中 | ✅ | 随 FunASR |
| Energy-based | 能量阈值 | 任意 | 低 | 极低 | ✅ | 无依赖 |

#### 7.2.2 选型决策

**✅ 选用 Silero VAD** 理由：
- 🎯 对噪声、远距离、弱声鲁棒性强
- ⚙️ 32ms 窗口与项目 CHUNK_SIZE=512 完美匹配（512/16000=32ms）
- 🔧 pip 一键安装，无需 Token 申请
- 🔩 三个关键参数均通过 .env 可配置

**❌ 未采用 WebRTC VAD**：能量算法，弱声/远距漏检率高。
**❌ 未采用 pyannote VAD**：需 HF Token + GPU，不支持流式，国内受限。
**❌ 未采用 FSMN-VAD**：无法独立实例化，多会话并发时状态混乱风险。

**双保险断句策略**：

```
触发条件 A（正常停顿）：静音持续 > VAD_SILENCE_DURATION_MS（默认 800ms）
触发条件 B（超长保护）：单段时长 > VAD_MAX_SEGMENT_DURATION（默认 30s）强制截断
两个条件任意满足其一即触发断句 → emit_segment()
```

---

### 7.3 说话人识别（Speaker Diarization）选型

#### 7.3.1 全市场方案横向对比

| 方案 | 开发方 | 嵌入维度 | 中文效果 | 速度 | 显存竞争 | 部署难度 |
|------|--------|---------|---------|------|---------|----------|
| ✅ **ERes2Net** | 达摩院 | 192 维 | 优 | 快 | 低（CPU 可跑）| modelscope 直下 |
| pyannote 3.1 | pyannote | 512 维 | 极优 | 中 | 中 | 需 HF Token |
| wespeaker | 西湖大学 | 256 维 | 优 | 快 | 低 | 公开可用 |
| ECAPA-TDNN | 各家 | 192 维 | 优 | 中 | 低 | 公开可用 |
| SpeakerNet | NVIDIA | 192 维 | 优 | 快 | 低 | NeMo 依赖 2GB+ |

#### 7.3.2 选型决策

**✅ 选用 ERes2Net** 理由：
- 🎯 针对中文会议场景优化，声纹分离准确
- 🔧 modelscope 直接下载，无需任何申请
- 💾 CPU 可运行，不与 ASR 争抢 GPU 显存
- 📦 已内置于 modelscope 生态，随项目环境安装

**❌ 未采用 pyannote 3.1**：需 HuggingFace Token 申请，国内网络访问受限。
**❌ 未采用 SpeakerNet**：NVIDIA NeMo 依赖树超 2GB，部署成本过高。
**❌ 未采用 wespeaker**：接口较新，社区资源少于 ERes2Net，稳定性待验证。

**在线聚类算法（OnlineSpeakerCluster，项目自实现）**：

```
新音频片段 → ERes2Net → 192 维声纹向量
    ↓
与历史所有声纹计算余弦相似度
    ↓
最高相似度 > 0.55 → 归入已有 speaker_N（同一人）
最高相似度 ≤ 0.55 → 新建 speaker_N+1（新说话人）
    ↓
查询 speaker_names 映射表
    ↓
返回 ("speaker_0", "张某某")
```

---

### 7.4 文本向量化（Milvus 语义检索）选型

#### 7.4.1 全市场方案横向对比

| 方案 | 向量维度 | 中文效果 | CPU 推理速度 | 显存占用 | 开源协议 |
|------|---------|---------|------------|---------|----------|
| ✅ **MiniLM-L12-v2 多语言** | 384 维 | 良好 | ~5ms/句 | 0（纯 CPU）| Apache2 |
| text2vec-base-chinese | 768 维 | 优秀 | ~10ms/句 | 0 | Apache2 |
| BGE-large-zh | 1024 维 | 极优 | ~20ms/句 | 0 | MIT |
| bce-embedding-base | 768 维 | 极优 | ~10ms/句 | 0 | Apache2 |
| OpenAI text-embedding-3 | 1536 维 | 极优 | 网络调用 | 0 | 付费 API |

#### 7.4.2 选型决策

**✅ 选用 paraphrase-multilingual-MiniLM-L12-v2** 理由：
- 💾 纯 CPU 推理，**完全不占 GPU 显存**，不与 ASR/声纹模型竞争
- ⚡ ~5ms/句，配合 fire-and-forget 异步写入，不影响实时推送延迟
- 🌍 支持 50+ 语言，适配未来多语言会议场景
- 🔧 Apache2 协议，商业可用

**❌ 未采用 BGE-large-zh**：1024 维对片段级检索过度，CPU 推理慢 4x。
**❌ 未采用 OpenAI Embedding**：需网络、付费，数据出内网，不符合本地部署要求。

---

### 7.5 模型资源占用总览

```
单卡 RTX 3090（24GB 显存）资源分配估算：

┌─────────────────────────────────────────────────────┐
│  paraformer-zh        2 GB  ████░░░░░░░░░░░░░░░░░   │
│  ERes2Net (CPU)       0 GB  （CPU 运行）             │
│  Silero VAD (CPU)     0 GB  （CPU 运行）             │
│  MiniLM (CPU)         0 GB  （CPU 运行）             │
│  系统/驱动            2 GB  ██░░░░░░░░░░░░░░░░░░░   │
│  剩余可用            20 GB  ← 支持 10 路并发会话    │
├─────────────────────────────────────────────────────┤
│  可选 Qwen3-ASR-1.7B  4 GB  ████░░░░░░░░░░░░░░░░░   │
│  可选 Whisper v3      6 GB  ██████░░░░░░░░░░░░░░░   │
└─────────────────────────────────────────────────────┘
```

---

## 八、项目结构说明

### 8.1 目录结构

```
speech_proj/
  backend/
    main.py                FastAPI 入口，lifespan 预加载所有模型
    core/
      config.py            pydantic-settings 统一配置，所有参数从 .env 读取
      logging.py           loguru，文件 + 控制台双输出
    db/
      database.py          asyncpg 异步引擎，AsyncSessionLocal
      models.py            3 张表，无 ForeignKey 无 relationship
    services/
      vad_service.py       Silero VAD，每会话独立实例
      asr_service.py       FunASR/Whisper/Qwen3 三后端，Semaphore 并发限流
      speaker_service.py   ERes2Net 声纹 + OnlineSpeakerCluster
      storage_service.py   MinIO 上传/下载/预签名 URL
      session_manager.py   SpeechSession + SessionManager 全局单例
      vector_service.py    MilvusService + TextEmbedder（可选）
    api/
      ws_handler.py        WebSocket 主处理器
      sessions.py          REST API 路由
    migrations/            Alembic 迁移脚本
  frontend/
    index.html             单页演示 UI
    app.js                 WebSocket 录音 + 波形可视化 + 说话人编辑
  docker/
    docker-compose.yml     PG + MinIO + Milvus + Backend
    Dockerfile.backend     CUDA 11.8 + Python 3.11
  scripts/                 工具脚本
  ARCHITECTURE.md          本文档
  README.md
```

### 8.2 Services 层设计原则

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个 Service 只做一件事，边界清晰 |
| **全局单例** | 模型加载耗时 5~30s，lifespan 预加载一次，所有请求复用 |
| **线程安全** | 推理均为同步阻塞，通过 asyncio.to_thread 派发线程池 |
| **热切换** | ASR 后端通过 .env 切换，无需改代码重新部署 |

| Service | 职责 | 单例 |
|---------|------|------|
| VADService | 语音端点检测，每会话独立实例 | ❌（每会话 new）|
| ASRService | 文本转写，Semaphore 并发限流 | ✅ |
| SpeakerService | 声纹提取 + 聚类 | ✅ |
| StorageService | MinIO 文件存取 | ✅ |
| MilvusService | 向量存取（可选）| ✅ |
| SessionManager | 会话生命周期管理 | ✅ |

### 8.3 SessionManager 设计

```
SessionManager（全局单例）
  sessions: Dict[session_id → SpeechSession]
  定期清理超时空闲会话（默认 600s）

SpeechSession（每个 WebSocket 连接独立一个）
  vad             独立 VAD 状态机，不同会话互不干扰
  cluster         独立说话人聚类，不同会话不混淆
  _raw_audio_chunks  原始 PCM 全量累积，会话结束打包 WAV 上传
  segments        已完成转写片段列表（内存）
  speaker_names   speaker_label → display_name 映射
  db_session_id   对应 PG 中的 sessions.id
```

---

## 九、关键配置速查

### 9.1 ASR 相关

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `ASR_BACKEND` | funasr | funasr / whisper / qwen3 |
| `ASR_MAX_CONCURRENT` | 2 | GPU 并发推理上限，防 OOM |
| `WHISPER_MODEL_SIZE` | large-v3 | Whisper 模型大小 |
| `WHISPER_COMPUTE_TYPE` | float16 | GPU:float16 / CPU:int8 |
| `QWEN3_ASR_MODEL_ID` | Qwen/Qwen3-ASR-1.7B | ModelScope 模型 ID |
| `QWEN3_ASR_MODEL_PATH` | 空 | 本地模型路径，空=自动下载 |
| `QWEN3_ASR_SYSTEM_PROMPT` | 见 .env.example | 控制输出风格/术语 |

### 9.2 VAD 相关

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `VAD_THRESHOLD` | 0.5 | 语音概率阈值，越高越严格 |
| `VAD_SILENCE_DURATION_MS` | 800 | 静音断句阈值（ms）|
| `VAD_MAX_SEGMENT_DURATION` | 30 | 最大片段时长（s）|

### 9.3 存储相关

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DATABASE_URL` | postgresql+asyncpg://... | PG 连接串 |
| `MINIO_ENDPOINT` | localhost:9000 | MinIO 地址 |
| `MINIO_ACCESS_KEY` | minioadmin | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | minioadmin | MinIO 秘密密钥 |
| `ENABLE_VECTOR_STORE` | false | 是否启用 Milvus |
| `MILVUS_HOST` | localhost | Milvus 地址 |

---

## 十、快速启动指南

```bash
# 1. 启动基础设施
cd docker
docker-compose up -d postgres minio

# 2. 初始化数据库
cd backend
alembic upgrade head

# 3. 配置环境变量
cp .env.example .env
# 按需编辑 .env（数据库密码、模型路径、ASR_BACKEND 等）

# 4. 启动后端服务
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 5.（可选）启动 Milvus 向量库
docker-compose up -d milvus
# 在 .env 设置 ENABLE_VECTOR_STORE=true 后重启后端

# 6. 打开演示前端
# 浏览器访问 frontend/index.html
```

### 切换 ASR 后端示例

```bash
# 切换到 Qwen3-ASR（中文最高精度）
echo "ASR_BACKEND=qwen3" >> .env
# 重启后端，模型自动从 ModelScope 下载

# 切换到 Whisper（多语言）
echo "ASR_BACKEND=whisper" >> .env
```

---

*文档结束 — ARCHITECTURE.md v3.0*
