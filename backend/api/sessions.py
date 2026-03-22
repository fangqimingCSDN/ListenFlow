"""
会话管理 REST API (续)

端点:
  GET    /api/sessions/{id}/download   获取下载预签名URL
  POST   /api/sessions/{id}/stop       停止录音
  POST   /api/sessions/{id}/pause      暂停
  POST   /api/sessions/{id}/resume     恢复
  DELETE /api/sessions/{id}            删除会话
"""
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from ..services.session_manager import session_manager
from ..services.storage_service import get_storage_service
from ..db.database import AsyncSessionLocal
from ..db import models
from sqlalchemy import select

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SpeakerNameUpdate(BaseModel):
    mapping: Dict[str, str]


def _session_or_404(session_id: str):
    session = session_manager._sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在或已过期")
    return session


@router.get("")
async def list_sessions():
    return {"sessions": session_manager.list_sessions()}


@router.get("/{session_id}")
async def get_session(session_id: str):
    session = _session_or_404(session_id)
    return {
        "session_id": session_id,
        "is_recording": session.is_recording,
        "is_paused": session.is_paused,
        "segment_count": session.segment_seq,
        "duration_sec": round(session.duration_sec(), 2),
        "speakers": session.get_speakers(),
        "speaker_count": session.cluster.get_speaker_count(),
        "last_active": session.last_active,
    }


@router.get("/{session_id}/speakers")
async def get_speakers(session_id: str):
    session = _session_or_404(session_id)
    return {"session_id": session_id, "speakers": session.get_speakers()}


@router.put("/{session_id}/speakers")
async def update_speakers(session_id: str, body: SpeakerNameUpdate):
    """
    编辑说话人名称映射。
    请求体: {"mapping": {"speaker_0": "张某某", "speaker_1": "李某某"}}
    """
    session = _session_or_404(session_id)
    session.update_speaker_names(body.mapping)
    if session.db_session_id:
        try:
            async with AsyncSessionLocal() as db:
                for label, name in body.mapping.items():
                    result = await db.execute(
                        select(models.Speaker).where(
                            models.Speaker.session_id == session.db_session_id,
                            models.Speaker.speaker_label == label,
                        )
                    )
                    row = result.scalar_one_or_none()
                    if row:
                        row.display_name = name
                await db.commit()
        except Exception as e:
            logger.warning(f"[DB] 说话人名称更新失败: {e}")
    return {"status": "ok", "session_id": session_id, "speakers": session.get_speakers()}


@router.get("/{session_id}/transcript")
async def get_transcript(session_id: str, named: bool = True):
    """获取当前转写文本 (?named=true 使用真实姓名)"""
    session = _session_or_404(session_id)
    return {
        "session_id": session_id,
        "segment_count": session.segment_seq,
        "text": session.build_transcript_text(use_display_name=named),
        "structured": session.build_transcript_json(),
    }


@router.get("/{session_id}/download")
async def get_download_urls(session_id: str, expires_hours: int = 24):
    """
    获取音频和文本文件的 MinIO 预签名下载URL（默认24小时有效）。
    前端可直接用返回的URL下载，无需经过后端。
    """
    session = _session_or_404(session_id)
    storage = get_storage_service()
    urls: Dict[str, str] = {}

    if session.audio_object_key:
        try:
            urls["audio_url"] = await storage.get_audio_download_url(
                session.audio_object_key, expires_hours
            )
        except Exception as e:
            logger.warning(f"获取音频URL失败: {e}")
            urls["audio_url"] = ""
    else:
        urls["audio_url"] = ""

    if session.transcript_object_key:
        try:
            urls["transcript_url"] = await storage.get_text_download_url(
                session.transcript_object_key, expires_hours
            )
        except Exception as e:
            logger.warning(f"获取文本URL失败: {e}")
            urls["transcript_url"] = ""
    else:
        urls["transcript_url"] = ""

    return {
        "session_id": session_id,
        "expires_hours": expires_hours,
        **urls,
    }


@router.post("/{session_id}/stop")
async def stop_session(session_id: str):
    """停止录音，触发 MinIO 上传和 DB 更新"""
    session = _session_or_404(session_id)
    if not session.is_recording:
        return {"status": "already_stopped", "session_id": session_id}
    session.is_recording = False
    # 触发后台完成流程
    import asyncio
    from ..api.ws_handler import _finalize_session
    asyncio.create_task(_finalize_session(session))
    return {
        "status": "stopped",
        "session_id": session_id,
        "segment_count": session.segment_seq,
    }


@router.post("/{session_id}/pause")
async def pause_session(session_id: str):
    session = _session_or_404(session_id)
    if not session.is_recording or session.is_paused:
        return {"status": "not_changed", "is_paused": session.is_paused}
    session.is_paused = True
    return {"status": "paused", "session_id": session_id}


@router.post("/{session_id}/resume")
async def resume_session(session_id: str):
    session = _session_or_404(session_id)
    if not session.is_paused:
        return {"status": "not_changed", "is_paused": session.is_paused}
    session.is_paused = False
    return {"status": "resumed", "session_id": session_id}


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """从内存中移除会话（不删除 MinIO/DB 数据）"""
    session = session_manager._sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    await session_manager.remove_session(session_id)
    return {"status": "deleted", "session_id": session_id}
