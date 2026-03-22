"""
全局配置管理 - 使用 pydantic-settings 统一管理环境变量
"""
from functools import lru_cache
from typing import List, Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # -- 服务基础配置 --
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: Literal["development", "production", "testing"] = "development"
    secret_key: str = "change-me-in-production"

    # -- 数据库 --
    database_url: str = "postgresql+asyncpg://speech_user:password@localhost:5432/speech_db"

    # -- MinIO --
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket_audio: str = "speech-audio"
    minio_bucket_text: str = "speech-text"
    minio_secure: bool = False

    # -- Qwen3-ASR 配置 (asr_backend=qwen3 时生效) --
    # ModelScope 模型 ID（国内镜像，无需 HF token）
    qwen3_asr_model_id: str = "Qwen/Qwen3-ASR-1.7B"
    # 本地模型路径（已下载则填此路径，空=自动下载）
    qwen3_asr_model_path: str = ""
    # system prompt，控制输出语言/格式/术语风格
    qwen3_asr_system_prompt: str = "Transcribe the speech into Simplified Chinese text with punctuation."

    # -- ASR 后端选择 --
    # "funasr"  : FunASR paraformer-zh (中文最优，速度快，无需HF token)
    # "whisper" : faster-whisper large-v3 (多语言，精度高)
    asr_backend: str = "funasr"
    # 最大并发 ASR 转写数 (GPU 建议 1~2，防显存溢出; CPU 可调大)
    asr_max_concurrent: int = 2

    # faster-whisper 参数 (asr_backend=whisper 时生效)
    whisper_model_size: str = "large-v3"
    whisper_device: str = "auto"           # auto / cuda / cpu
    whisper_compute_type: str = "float16"  # float16(GPU) / int8(CPU)
    whisper_model_path: str = ""           # 空=自动下载，填路径=离线

    # -- 说话人识别 --
    huggingface_token: str = ""

    # -- VAD 配置 --
    vad_threshold: float = 0.5  # VAD语音概率阈値
    vad_silence_duration_ms: int = 800   # 静音超过此值(ms)则断句
    vad_max_segment_duration: int = 30   # 单段最长秒数，超过强制截断

    # -- 向量库 (可选) --
    enable_vector_store: bool = False
    # vector_backend: "milvus" | "qdrant"
    vector_backend: str = "milvus"

    # Milvus (推荐：生产级，支持十亿级向量，分布式)
    milvus_host: str = "localhost"
    milvus_port: int = 19530

    # Qdrant (备选：轻量单机)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "speech_transcripts"

    # -- CORS --
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # -- 日志 --
    log_level: str = "INFO"
    log_file: str = "logs/speech_proj.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """单例模式获取配置，避免重复加载"""
    return Settings()


settings = get_settings()
