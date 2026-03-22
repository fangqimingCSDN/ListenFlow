"""
数据库初始化脚本
运行: python scripts/init_db.py
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.database import init_db
from backend.core.logging import setup_logging


async def main():
    setup_logging()
    print("初始化数据库表结构...")
    await init_db()
    print("完成！")


if __name__ == "__main__":
    asyncio.run(main())
