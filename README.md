# 实时语音转写系统

基于 **FunASR + ERes2Net + Silero VAD** 的工程级实时语音识别与说话人分离服务。

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| ASR | FunASR paraformer-zh | 中文SOTA，非自回归极速推理，支持GPU/CPU |
| 标点恢复 | ct-punc-c | 辅助断句，输出自然标点 |
| VAD | Silero VAD | 轻量流式，32ms延迟窗口 |
| 说话人识别 | ERes2Net (iic) | 阿里达摩院声纹提取，余弦聚类 |
| 传输协议 | WebSocket | JSON消息协议，base64音频流 |
| 关系数据库 | PostgreSQL + SQLAlchemy | 异步asyncpg驱动 |
| 对象存储 | MinIO | 原始音频 + 转写文本，预签名URL下载 |
| 向量库(可选) | Qdrant | docker compose --profile vector up |
| Web框架 | FastAPI + uvicorn | 单worker（AI模型不支持fork多进程）|

## 项目结构

```
speech_proj/
├── backend/
│   ├── main.py                  # FastAPI 应用入口
│   ├── requirements.txt
│   ├── .env.example             # 环境变量配置模板
│   ├── alembic.ini
│   ├── core/
│   │   ├── config.py            # pydantic-settings 配置
│   │   └── logging.py           # loguru 日志
│   ├── db/
│   │   ├── database.py          # 异步 SQLAlchemy 引擎
│   │   └── models.py            # Session / Speaker / TranscriptSegment
│   ├── services/
│   │   ├── vad_service.py       # Silero VAD 流式封装
│   │   ├── asr_service.py       # FunASR paraformer-zh
│   │   ├── speaker_service.py   # ERes2Net + 在线聚类
│   │   ├── storage_service.py   # MinIO 上传/下载/预签名URL
│   │   └── session_manager.py   # 会话生命周期管理
│   ├── api/
│   │   ├── ws_handler.py        # WebSocket 主处理器
│   │   └── sessions.py          # REST API 路由
│   └── migrations/              # Alembic 数据库迁移
├── frontend/
│   ├── index.html               # 单页演示前端
│   └── app.js                   # WebSocket + 录音逻辑
├── docker/
│   ├── docker-compose.yml       # PG + MinIO + Qdrant + Backend
│   └── Dockerfile.backend
└── scripts/
    └── init_db.py
```

## 快速启动

### 1. 启动基础服务（PostgreSQL + MinIO）

```bash
cd docker
docker compose up -d postgres minio
```

### 2. 配置后端环境

```bash
cd backend
copy .env.example .env
# 编辑 .env，填写实际配置
pip install -r requirements.txt
```

### 3. 初始化数据库

```bash
python ../scripts/init_db.py
```

### 4. 启动后端

```bash
# 在项目根目录运行
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 打开前端

用浏览器直接打开 `frontend/index.html`，或通过 HTTP 服务器托管。

```bash
python -m http.server 3000 --directory frontend
# 访问 http://localhost:3000
```

### 6. 全量 Docker 部署

```bash
cd docker
docker compose up -d
# 带向量库:
docker compose --profile vector up -d
```

## WebSocket 协议

连接地址: `ws://localhost:8000/ws/speech`

**客户端 → 服务端**

```json
{"type": "start", "session_id": "uuid"}    // 开始会话
{"type": "audio", "data": "<base64 PCM>"}  // 发送16bit PCM音频块
{"type": "pause"}                           // 暂停
{"type": "resume"}                          // 恢复
{"type": "stop"}                            // 停止并完成
```

**服务端 → 客户端**

```json
{"type": "session_started", "session_id": "uuid"}
{"type": "transcript",
  "seq": 1, "text": "你好世界",
  "speaker_label": "speaker_0",
  "speaker_display": "张某某",
  "start": 1.23, "end": 3.45}
{"type": "session_completed",
  "audio_url": "<MinIO预签名URL>",
  "transcript_url": "<MinIO预签名URL>"}
```

## REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| GET | /api/sessions | 列出活跃会话 |
| GET | /api/sessions/{id} | 会话详情 |
| GET | /api/sessions/{id}/speakers | 获取说话人列表 |
| **PUT** | /api/sessions/{id}/speakers | **编辑说话人姓名** |
| GET | /api/sessions/{id}/transcript | 获取转写文本 |
| GET | /api/sessions/{id}/download | 获取下载预签名URL |
| POST | /api/sessions/{id}/stop | 停止录音 |
| POST | /api/sessions/{id}/pause | 暂停 |
| POST | /api/sessions/{id}/resume | 恢复 |
| DELETE | /api/sessions/{id} | 删除会话 |

交互式文档: http://localhost:8000/docs

## 说话人姓名编辑

```bash
# 将 speaker_0 命名为张某某，speaker_1 命名为李某某
curl -X PUT http://localhost:8000/api/sessions/{session_id}/speakers \
  -H 'Content-Type: application/json' \
  -d '{"mapping": {"speaker_0": "张某某", "speaker_1": "李某某"}}'
```

实时效果：前端转写区已渲染的片段名称立即更新。

## 断句策略

1. **VAD静音断句**：Silero VAD 检测到语音结束且静音超过 `VAD_SILENCE_DURATION_MS`（默认800ms）
2. **最大时长断句**：单段超过 `VAD_MAX_SEGMENT_DURATION`（默认30秒）强制截断
3. **标点辅助**：ct-punc-c 为文本恢复标点，配合前端可视化句子边界

可在 `.env` 中调整：
```
VAD_SILENCE_DURATION_MS=800   # 静音阈值（毫秒）
VAD_MAX_SEGMENT_DURATION=30   # 最大片段时长（秒）
VAD_THRESHOLD=0.5             # VAD语音概率阈值
```

## 模型选型说明（可替换方案）

| 方案 | ASR | 说话人 | 适用场景 |
|------|-----|--------|----------|
| **A（当前）** | paraformer-zh | ERes2Net | 中文为主，速度优先 |
| B | faster-whisper large-v3 | pyannote 3.1 | 多语言，精度优先（需HF Token）|
| C | SenseVoice | ERes2Net | 多语言+情感识别 |

替换 ASR 只需修改 `backend/services/asr_service.py` 中的模型加载逻辑。

## 存储说明

- **原始音频**：会话结束后打包为 WAV 上传到 MinIO `speech-audio/{session_id}/raw_audio.wav`
- **转写文本**：上传到 MinIO `speech-text/{session_id}/transcript.txt`（含说话人姓名）
- **结构化数据**：同步写入 PostgreSQL，支持查询和统计
- **下载方式**：通过预签名 URL 直接从 MinIO 下载，无需经过后端
