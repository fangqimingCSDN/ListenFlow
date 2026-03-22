"""
FastAPI 应用入口

启动命令:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

健康检查:
  GET /health

WebSocket:
  ws://localhost:8000/ws/speech

REST API 文档:
  http://localhost:8000/docs
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from .core.config import settings
from .core.logging import setup_logging
from .db.database import init_db
from .services.asr_service import get_asr_service
from .services.speaker_service import get_speaker_service
from .services.storage_service import get_storage_service
from .services.session_manager import session_manager
from .api.sessions import router as sessions_router
from .api.ws_handler import websocket_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理:
      启动时: 初始化日志、DB、预加载AI模型、连接MinIO
      关闭时: 清理资源
    """
    # ── 启动 ──────────────────────────────────────────────────────────────
    setup_logging()
    logger.info("=" * 60)
    logger.info("实时语音转写系统 启动中...")
    logger.info(f"环境: {settings.app_env}")

    # 初始化数据库表
    try:
        await init_db()
        logger.info("PostgreSQL 初始化完成")
    except Exception as e:
        logger.error(f"PostgreSQL 初始化失败: {e}")

    # 预加载 AI 模型（启动时一次性加载，避免首次请求延迟）
    logger.info("预加载 AI 模型（ASR + 声纹）...")
    try:
        # 在线程池中加载，不阻塞事件循环
        await asyncio.to_thread(get_asr_service)
        logger.info("ASR 模型加载完成")
    except Exception as e:
        logger.error(f"ASR 模型加载失败: {e}")

    try:
        await asyncio.to_thread(get_speaker_service)
        logger.info("声纹模型加载完成")
    except Exception as e:
        logger.error(f"声纹模型加载失败: {e}")

    # 连接 MinIO
    try:
        get_storage_service()
        logger.info("MinIO 连接完成")
    except Exception as e:
        logger.error(f"MinIO 连接失败: {e}")

    # 启动空闲会话清理后台任务
    cleanup_task = asyncio.create_task(_cleanup_loop())

    logger.info("系统启动完成，等待连接...")
    logger.info("=" * 60)

    yield  # 应用运行中

    # ── 关闭 ──────────────────────────────────────────────────────────────
    cleanup_task.cancel()
    logger.info("系统正在关闭...")


async def _cleanup_loop():
    """每5分钟清理一次空闲超时会话"""
    while True:
        await asyncio.sleep(300)
        try:
            await session_manager.cleanup_idle()
        except Exception as e:
            logger.error(f"清理任务异常: {e}")


# ── 创建 FastAPI 应用 ─────────────────────────────────────────────────────────
app = FastAPI(
    title="实时语音转写系统",
    description="基于 FunASR + ERes2Net + Silero VAD 的实时语音识别与说话人分离服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由注册 ──────────────────────────────────────────────────────────────────

# REST API
app.include_router(sessions_router)

# WebSocket
@app.websocket("/ws/speech")
async def ws_speech(websocket: WebSocket):
    """实时语音识别 WebSocket 端点"""
    await websocket_handler(websocket)


# 健康检查
@app.get("/health", tags=["system"])
async def health_check():
    import torch
    return {
        "status": "healthy",
        "env": settings.app_env,
        "cuda_available": torch.cuda.is_available(),
        "active_sessions": len(session_manager._sessions),
        "whisper_model": settings.whisper_model_size,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        workers=1,  # AI模型不支持多进程，使用单worker
        log_level=settings.log_level.lower(),
    )
