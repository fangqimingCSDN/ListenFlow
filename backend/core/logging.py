"""
日志配置 - 使用 loguru 统一日志格式
"""
import sys
import os
from loguru import logger
from .config import settings


def setup_logging():
    """初始化日志配置"""
    os.makedirs("logs", exist_ok=True)

    # 移除默认handler
    logger.remove()

    # 控制台输出（带颜色）
    logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # 文件输出（按天滚动，保留30天）
    logger.add(
        settings.log_file,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    return logger
