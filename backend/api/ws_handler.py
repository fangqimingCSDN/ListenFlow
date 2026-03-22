import asyncio
import base64
import io
import json
import uuid
from typing import Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from ..services.session_manager import session_manager, SpeechSession
from ..services.asr_service import get_asr_service
from ..services.speaker_service import get_speaker_service
from ..services.storage_service import get_storage_service
from ..services.vector_service import get_milvus_service
from ..services.vad_service import SAMPLE_RATE
from ..db.database import AsyncSessionLocal
from ..db import models


async def _send(ws: WebSocket, payload: dict):
    try:
        await ws.send_json(payload)
    except Exception:
        pass


async def _process_segment(
    ws: WebSocket,
    speech_session: SpeechSession,
    start_time: float,
    end_time: float,
    audio: np.ndarray,
    upload_segments: bool = False,
):
    """
    处理一段经过 VAD 端点检测后的完整语音片段。
    为何该操作不会阻塞 WebSocket 接收循环：
    通过 asyncio.create_task() 调用 → 立即返回
    语音识别（ASR）和说话人嵌入提取通过 asyncio.to_thread 在线程池中运行
    事件循环始终空闲，可持续接收新的音频数据包
    语音识别服务（ASRService）中的信号量限制了并发的 GPU 推理数量
    """
    asr_service = get_asr_service()
    speaker_service = get_speaker_service()
    storage_service = get_storage_service()

    # ASR transcription + speaker embedding in parallel (both thread-pool)
    asr_result, embedding = await asyncio.gather(
        asr_service.transcribe(audio, start_time, end_time),
        speaker_service.extract_embedding(audio),
    )

    if asr_result is None or not asr_result.text.strip():
        return

    if embedding is not None:
        speaker_label, speaker_display = speech_session.identify_speaker(embedding)
    else:
        speaker_label = "speaker_0"
        speaker_display = speech_session.speaker_names.get("speaker_0", "speaker_0")

    logger.info(f"[WS][{speech_session.session_id[:8]}] {speaker_display}: {asr_result.text[:60]}")

    audio_object_key = ""
    if upload_segments:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        audio_object_key = await storage_service.upload_audio_segment(
            speech_session.session_id, speech_session.segment_seq + 1, buf.getvalue()
        )

    seg = speech_session.add_segment(
        start_time=start_time, end_time=end_time,
        text=asr_result.text, speaker_label=speaker_label,
        speaker_display=speaker_display, confidence=asr_result.confidence,
        audio_object_key=audio_object_key,
    )

    # fire-and-forget: PostgreSQL write
    if speech_session.db_session_id:
        asyncio.create_task(_write_segment_to_db(speech_session, seg))

    # fire-and-forget: Milvus vector write (if enabled)
    milvus = get_milvus_service()
    if milvus is not None:
        asyncio.create_task(milvus.insert_segment(
            session_id=speech_session.session_id,
            speaker_label=seg.speaker_label,
            speaker_display=seg.speaker_display,
            start_time=seg.start_time, end_time=seg.end_time,
            text=seg.text,
        ))

    # push to WebSocket client immediately
    await _send(ws, {
        "type": "transcript", "seq": seg.seq,
        "text": seg.text, "speaker_label": seg.speaker_label,
        "speaker_display": seg.speaker_display,
        "start": round(seg.start_time, 3), "end": round(seg.end_time, 3),
        "duration": round(seg.end_time - seg.start_time, 3),
        "confidence": seg.confidence,
    })


async def _write_segment_to_db(speech_session: SpeechSession, seg):
    try:
        from sqlalchemy import select
        import datetime
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(models.Speaker).where(
                models.Speaker.session_id == speech_session.db_session_id,
                models.Speaker.speaker_label == seg.speaker_label,
            ))
            speaker_row = result.scalar_one_or_none()
            if speaker_row is None:
                speaker_row = models.Speaker(
                    session_id=speech_session.db_session_id,
                    speaker_label=seg.speaker_label,
                    display_name=seg.speaker_display if seg.speaker_display != seg.speaker_label else None,
                )
                db.add(speaker_row)
                await db.flush()
            db.add(models.TranscriptSegment(
                session_id=speech_session.db_session_id, speaker_id=speaker_row.id,
                start_time=seg.start_time, end_time=seg.end_time,
                text=seg.text, confidence=seg.confidence,
                sequence_no=seg.seq, is_final=True,
            ))
            await db.commit()
    except Exception as e:
        logger.error(f"[DB] write segment failed: {e}")

async def _finalize_session(speech_session: SpeechSession):
    import datetime
    from sqlalchemy import select
    storage = get_storage_service()
    sid = speech_session.session_id
    try:
        wav_bytes = speech_session.get_raw_audio_wav()
        if wav_bytes:
            speech_session.audio_object_key = await storage.upload_audio(
                sid, wav_bytes, object_name=f"{sid}/raw_audio.wav", content_type="audio/wav"
            )
    except Exception as e:
        logger.error(f"[Finalize] audio upload failed: {e}")
    try:
        transcript_text = speech_session.build_transcript_text(use_display_name=True)
        speech_session.transcript_object_key = await storage.upload_transcript(sid, transcript_text, named=False)
        await storage.upload_transcript_json(sid, speech_session.build_transcript_json())
    except Exception as e:
        logger.error(f"[Finalize] text upload failed: {e}")
    if speech_session.db_session_id:
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(models.Session).where(models.Session.id == speech_session.db_session_id)
                )
                row = result.scalar_one_or_none()
                if row:
                    row.status = "completed"
                    row.audio_object_key = speech_session.audio_object_key
                    row.transcript_object_key = speech_session.transcript_object_key
                    row.audio_duration_sec = speech_session.duration_sec()
                    row.speaker_count = speech_session.cluster.get_speaker_count()
                    row.completed_at = datetime.datetime.utcnow()
                    await db.commit()
        except Exception as e:
            logger.error(f"[DB] session update failed: {e}")
    logger.info(f"[Finalize] session {sid} completed")


async def websocket_handler(websocket: WebSocket):
    await websocket.accept()
    speech_session: Optional[SpeechSession] = None
    logger.info("[WS] new connection")
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _send(websocket, {"type": "error", "message": "JSON required"})
                continue
            msg_type = msg.get("type")

            if msg_type == "start":
                session_id = msg.get("session_id") or str(uuid.uuid4())
                speech_session = await session_manager.get_or_create(session_id)
                speech_session.is_recording = True
                speech_session.is_paused = False
                try:
                    async with AsyncSessionLocal() as db:
                        db_sess = models.Session(
                            id=uuid.uuid4(),
                            title=msg.get("title", f"rec_{session_id[:8]}"),
                            status="recording",
                            language=msg.get("language", "zh"),
                        )
                        db.add(db_sess)
                        await db.commit()
                        speech_session.db_session_id = db_sess.id
                except Exception as e:
                    logger.warning(f"[DB] session create failed (recording ok): {e}")
                await _send(websocket, {"type": "session_started", "session_id": session_id})
                logger.info(f"[WS] session started: {session_id}")

            elif msg_type == "audio":
                if speech_session is None:
                    await _send(websocket, {"type": "error", "message": "send start first"})
                    continue
                try:
                    audio_bytes = base64.b64decode(msg["data"])
                except Exception:
                    await _send(websocket, {"type": "error", "message": "base64 decode failed"})
                    continue
                # create_task = schedule segment processing, return immediately
                # receive loop stays unblocked regardless of how long ASR takes
                for (s, e, audio_np) in speech_session.push_audio(audio_bytes):
                    asyncio.create_task(
                        _process_segment(websocket, speech_session, s, e, audio_np)
                    )

            elif msg_type == "pause":
                if speech_session:
                    speech_session.is_paused = True
                    await _send(websocket, {"type": "paused"})

            elif msg_type == "resume":
                if speech_session:
                    speech_session.is_paused = False
                    await _send(websocket, {"type": "resumed"})

            elif msg_type == "stop":
                if speech_session:
                    speech_session.is_recording = False
                    # FIX: use create_task here too so stop response is not delayed
                    for (s, e, audio_np) in speech_session.flush_vad():
                        asyncio.create_task(
                            _process_segment(websocket, speech_session, s, e, audio_np)
                        )
                    asyncio.create_task(_finalize_session(speech_session))
                    storage = get_storage_service()
                    audio_url, transcript_url = "", ""
                    try:
                        if speech_session.audio_object_key:
                            audio_url = await storage.get_audio_download_url(speech_session.audio_object_key)
                        if speech_session.transcript_object_key:
                            transcript_url = await storage.get_text_download_url(speech_session.transcript_object_key)
                    except Exception:
                        pass
                    await _send(websocket, {
                        "type": "session_completed",
                        "session_id": speech_session.session_id,
                        "segment_count": speech_session.segment_seq,
                        "speaker_count": speech_session.cluster.get_speaker_count(),
                        "duration_sec": round(speech_session.duration_sec(), 2),
                        "audio_url": audio_url,
                        "transcript_url": transcript_url,
                    })
                    logger.info(f"[WS] session stopped: {speech_session.session_id}")

            else:
                await _send(websocket, {"type": "error", "message": f"unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info("[WS] client disconnected")
        if speech_session and speech_session.is_recording:
            speech_session.is_recording = False
            asyncio.create_task(_finalize_session(speech_session))
    except Exception as e:
        logger.error(f"[WS] unhandled error: {e}")
        await _send(websocket, {"type": "error", "message": str(e)})
