# -*- coding: utf-8 -*-
import pathlib

SEC7 = '''## 八、项目结构说明

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
'''

pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/scripts/_arch_sec7.txt').write_text(SEC7, encoding='utf-8')
print('sec7 len:', len(SEC7))
print('done')
