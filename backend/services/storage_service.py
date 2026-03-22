"""
MinIO 对象存储服务

存储结构:
  Bucket: speech-audio
    {session_id}/raw_audio.webm        # 完整原始录音（流式追加）
    {session_id}/segments/{seq:04d}.wav  # 各断句片段音频（可选）

  Bucket: speech-text
    {session_id}/transcript.txt        # 纯文本转写（说话人 + 内容）
    {session_id}/transcript.json       # 带时间戳结构化数据
    {session_id}/named_transcript.txt  # 含真实姓名的版本

所有上传操作通过 asyncio.to_thread 异步化，不阻塞 WebSocket 主循环。
"""
import io
import json
import asyncio
from typing import Optional
from datetime import timedelta
from loguru import logger
from minio import Minio
from minio.error import S3Error
from ..core.config import settings


class StorageService:
    """MinIO 存储服务封装（全局单例）"""

    def __init__(self):
        self._client: Optional[Minio] = None
        self._connect()
        self._ensure_buckets()

    def _connect(self):
        try:
            self._client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
            logger.info(f"MinIO 连接成功: {settings.minio_endpoint}")
        except Exception as e:
            logger.error(f"MinIO 连接失败: {e}")
            raise

    def _ensure_buckets(self):
        """确保必要的 Bucket 存在，不存在则自动创建"""
        for bucket in [settings.minio_bucket_audio, settings.minio_bucket_text]:
            try:
                if not self._client.bucket_exists(bucket):
                    self._client.make_bucket(bucket)
                    logger.info(f"创建 MinIO Bucket: {bucket}")
            except S3Error as e:
                logger.error(f"Bucket操作失败 [{bucket}]: {e}")
                raise

    # ── 音频上传 ─────────────────────────────────────────────────────────────

    def upload_audio_sync(
        self,
        session_id: str,
        data: bytes,
        object_name: Optional[str] = None,
        content_type: str = "audio/wav",
    ) -> str:
        """
        上传音频到 MinIO。

        Returns:
            MinIO 对象 Key
        """
        ext = content_type.split("/")[-1]
        if object_name is None:
            object_name = f"{session_id}/raw_audio.{ext}"

        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=settings.minio_bucket_audio,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        logger.info(
            f"[Storage] 音频上传: {settings.minio_bucket_audio}/{object_name} "
            f"({len(data) / 1024:.1f} KB)"
        )
        return object_name

    async def upload_audio(
        self,
        session_id: str,
        data: bytes,
        object_name: Optional[str] = None,
        content_type: str = "audio/wav",
    ) -> str:
        """异步上传完整会话音频"""
        return await asyncio.to_thread(
            self.upload_audio_sync, session_id, data, object_name, content_type
        )

    def upload_audio_segment_sync(
        self,
        session_id: str,
        seq: int,
        data: bytes,
        content_type: str = "audio/wav",
    ) -> str:
        """上传单个断句音频片段"""
        object_name = f"{session_id}/segments/{seq:04d}.wav"
        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=settings.minio_bucket_audio,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        return object_name

    async def upload_audio_segment(
        self, session_id: str, seq: int, data: bytes
    ) -> str:
        """异步上传断句音频片段"""
        return await asyncio.to_thread(
            self.upload_audio_segment_sync, session_id, seq, data
        )

    # ── 文本上传 ─────────────────────────────────────────────────────────────

    def upload_text_sync(
        self,
        session_id: str,
        content: str,
        object_name: str,
        content_type: str = "text/plain; charset=utf-8",
    ) -> str:
        """上传文本内容到 MinIO"""
        data = content.encode("utf-8")
        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=settings.minio_bucket_text,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        logger.info(
            f"[Storage] 文本上传: {settings.minio_bucket_text}/{object_name}"
        )
        return object_name

    async def upload_transcript(
        self, session_id: str, text_content: str, named: bool = False
    ) -> str:
        """异步上传转写文本（普通版或带姓名版）"""
        suffix = "named_" if named else ""
        object_name = f"{session_id}/{suffix}transcript.txt"
        return await asyncio.to_thread(
            self.upload_text_sync, session_id, text_content, object_name
        )

    async def upload_transcript_json(
        self, session_id: str, data: dict
    ) -> str:
        """异步上传结构化 JSON 转写数据"""
        object_name = f"{session_id}/transcript.json"
        content = json.dumps(data, ensure_ascii=False, indent=2)
        return await asyncio.to_thread(
            self.upload_text_sync,
            session_id,
            content,
            object_name,
            "application/json; charset=utf-8",
        )

    # ── 下载 / 预签名 URL ────────────────────────────────────────────────────

    def get_presigned_url_sync(
        self,
        bucket: str,
        object_name: str,
        expires_hours: int = 24,
    ) -> str:
        """
        生成预签名下载 URL（有效期默认24小时）。
        前端可直接用该 URL 下载文件，无需经过后端。
        """
        url = self._client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(hours=expires_hours),
        )
        return url

    async def get_audio_download_url(
        self, object_name: str, expires_hours: int = 24
    ) -> str:
        """获取音频文件预签名下载 URL"""
        return await asyncio.to_thread(
            self.get_presigned_url_sync,
            settings.minio_bucket_audio,
            object_name,
            expires_hours,
        )

    async def get_text_download_url(
        self, object_name: str, expires_hours: int = 24
    ) -> str:
        """获取文本文件预签名下载 URL"""
        return await asyncio.to_thread(
            self.get_presigned_url_sync,
            settings.minio_bucket_text,
            object_name,
            expires_hours,
        )

    def download_bytes_sync(self, bucket: str, object_name: str) -> bytes:
        """同步下载对象为字节"""
        response = self._client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    async def download_bytes(self, bucket: str, object_name: str) -> bytes:
        """异步下载对象为字节"""
        return await asyncio.to_thread(
            self.download_bytes_sync, bucket, object_name
        )

    def object_exists_sync(self, bucket: str, object_name: str) -> bool:
        """检查对象是否存在"""
        try:
            self._client.stat_object(bucket, object_name)
            return True
        except S3Error:
            return False


# ── 全局单例 ─────────────────────────────────────────────────────────────────
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
