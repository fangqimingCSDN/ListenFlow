"""
会话管理器

每个 WebSocket 连接对应一个 SpeechSession，负责:
  1. 维护该会话的 VAD / Speaker Cluster 独立状态
  2. 累积原始音频（用于会话结束后整体上传 MinIO）
  3. 追踪说话人映射（speaker_0 → 张某某）
  4. 写入 PostgreSQL（异步）

SessionManager 全局单例，管理所有活跃会话的生命周期。
"""
import asyncio
import time
import uuid
import io
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from loguru import logger

from .vad_service import VADService, SAMPLE_RATE
from .speaker_service import OnlineSpeakerCluster


@dataclass
class SegmentRecord:
    """一个识别完成的语音片段"""
    seq: int
    start_time: float
    end_time: float
    text: str
    speaker_label: str        # e.g. "speaker_0"
    speaker_display: str      # e.g. "张某某" 或同 speaker_label
    confidence: float = 1.0
    audio_object_key: str = ""  # MinIO 路径（可选）


class SpeechSession:
    """
    单个录音会话状态容器。
    由 WebSocket handler 创建，SessionManager 统一管理。
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = time.time()
        self.last_active = time.time()
        self.is_recording = True
        self.is_paused = False

        # 每个会话独立的 VAD 和说话人聚类
        self.vad = VADService()
        self.cluster = OnlineSpeakerCluster(thr=0.55, ema=0.9)

        # 说话人标签 → 真实姓名映射（用户可编辑）
        self.speaker_names: Dict[str, str] = {}   # {"speaker_0": "张某某"}

        # 识别结果序列
        self.segments: List[SegmentRecord] = []
        self.segment_seq: int = 0

        # 累积原始音频字节（用于整体上传 MinIO）
        self._raw_audio_chunks: List[bytes] = []
        self._total_audio_bytes: int = 0

        # MinIO 存储 Key（会话完成后填写）
        self.audio_object_key: Optional[str] = None
        self.transcript_object_key: Optional[str] = None

        # DB 会话 UUID
        self.db_session_id: Optional[uuid.UUID] = None

        logger.info(f"[Session] 新会话创建: {session_id}")

    # ── 音频输入 ──────────────────────────────────────────────────────────────

    def push_audio(self, audio_bytes: bytes) -> List:
        """
        接收原始音频字节，累积保存并喂给 VAD。
        返回 VAD 完成的语音片段列表: [(start_sec, end_sec, np.ndarray), ...]
        """
        if not self.is_recording or self.is_paused:
            return []
        self.last_active = time.time()
        self._raw_audio_chunks.append(audio_bytes)
        self._total_audio_bytes += len(audio_bytes)
        return self.vad.process_chunk(audio_bytes)

    def flush_vad(self) -> List:
        """会话停止时强制 VAD 输出最后片段"""
        return self.vad.flush()

    # ── 说话人 ────────────────────────────────────────────────────────────────

    def identify_speaker(self, embedding) -> tuple:
        """
        输入声纹嵌入，返回 (speaker_label, display_name)。
        e.g. ("speaker_0", "张某某")
        """
        speaker_id = self.cluster.update(embedding)
        label = f"speaker_{speaker_id}"
        display = self.speaker_names.get(label, label)
        return label, display

    def update_speaker_names(self, mapping: Dict[str, str]):
        """
        更新说话人显示名称，并回填历史片段。
        mapping: {"speaker_0": "张某某", "speaker_1": "李某某"}
        """
        self.speaker_names.update(mapping)
        for seg in self.segments:
            if seg.speaker_label in self.speaker_names:
                seg.speaker_display = self.speaker_names[seg.speaker_label]
        logger.info(f"[Session] {self.session_id} 说话人映射更新: {mapping}")

    def get_speakers(self) -> List[Dict]:
        """返回当前会话中识别到的所有说话人信息"""
        n = self.cluster.get_speaker_count()
        return [
            {
                "speaker_label": f"speaker_{i}",
                "display_name": self.speaker_names.get(f"speaker_{i}", f"speaker_{i}"),
            }
            for i in range(n)
        ]

    # ── 片段记录 ──────────────────────────────────────────────────────────────

    def add_segment(
        self,
        start_time: float,
        end_time: float,
        text: str,
        speaker_label: str,
        speaker_display: str,
        confidence: float = 1.0,
        audio_object_key: str = "",
    ) -> SegmentRecord:
        """添加一条识别结果片段"""
        self.segment_seq += 1
        seg = SegmentRecord(
            seq=self.segment_seq,
            start_time=start_time,
            end_time=end_time,
            text=text,
            speaker_label=speaker_label,
            speaker_display=speaker_display,
            confidence=confidence,
            audio_object_key=audio_object_key,
        )
        self.segments.append(seg)
        return seg

    # ── 音频导出 ──────────────────────────────────────────────────────────────

    def get_raw_audio_wav(self) -> bytes:
        """
        将累积的 PCM 音频打包为标准 WAV 文件字节，用于上传 MinIO。
        """
        import numpy as np
        import soundfile as sf

        if not self._raw_audio_chunks:
            return b""

        all_bytes = b"".join(self._raw_audio_chunks)
        try:
            audio = np.frombuffer(all_bytes, dtype=np.float32)
        except Exception:
            audio = np.frombuffer(all_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        buf = io.BytesIO()
        sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # ── 文本导出 ──────────────────────────────────────────────────────────────

    def build_transcript_text(self, use_display_name: bool = True) -> str:
        """生成纯文本转写，格式: 说话人\t文本"""
        lines = []
        for seg in self.segments:
            name = seg.speaker_display if use_display_name else seg.speaker_label
            lines.append(f"{name}\t{seg.text}")
        return "\n".join(lines)

    def build_transcript_json(self) -> dict:
        """生成结构化 JSON 转写数据"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "speakers": self.get_speakers(),
            "segments": [
                {
                    "seq": s.seq,
                    "start": round(s.start_time, 3),
                    "end": round(s.end_time, 3),
                    "duration": round(s.end_time - s.start_time, 3),
                    "speaker_label": s.speaker_label,
                    "speaker_display": s.speaker_display,
                    "text": s.text,
                    "confidence": s.confidence,
                    "audio_object_key": s.audio_object_key,
                }
                for s in self.segments
            ],
        }

    def duration_sec(self) -> float:
        """会话总录音时长（秒）"""
        return self._total_audio_bytes / (SAMPLE_RATE * 4)  # float32 = 4 bytes


# ── SessionManager（全局单例） ─────────────────────────────────────────────────

class SessionManager:
    """
    管理所有活跃 SpeechSession 的生命周期。
    - 创建 / 获取 / 删除会话
    - 后台定期清理超时会话
    """

    def __init__(self, idle_timeout_sec: int = 600):
        self._sessions: Dict[str, SpeechSession] = {}
        self._lock = asyncio.Lock()
        self._idle_timeout = idle_timeout_sec

    async def create_session(self, session_id: Optional[str] = None) -> SpeechSession:
        """创建新会话，返回 SpeechSession 实例"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        async with self._lock:
            session = SpeechSession(session_id)
            self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[SpeechSession]:
        """获取已有会话"""
        return self._sessions.get(session_id)

    async def get_or_create(self, session_id: str) -> SpeechSession:
        """获取或创建会话"""
        session = self._sessions.get(session_id)
        if session is None:
            session = await self.create_session(session_id)
        return session

    async def remove_session(self, session_id: str):
        """从内存中移除会话"""
        async with self._lock:
            self._sessions.pop(session_id, None)
        logger.info(f"[SessionManager] 会话已移除: {session_id}")

    def list_sessions(self) -> List[Dict]:
        """列出所有活跃会话摘要"""
        return [
            {
                "session_id": sid,
                "is_recording": s.is_recording,
                "is_paused": s.is_paused,
                "segment_count": s.segment_seq,
                "speaker_count": s.cluster.get_speaker_count(),
                "duration_sec": round(s.duration_sec(), 2),
                "last_active": s.last_active,
            }
            for sid, s in self._sessions.items()
        ]

    async def cleanup_idle(self):
        """清理超时空闲会话（由后台任务调用）"""
        now = time.time()
        to_remove = [
            sid for sid, s in self._sessions.items()
            if now - s.last_active > self._idle_timeout and not s.is_recording
        ]
        for sid in to_remove:
            await self.remove_session(sid)
            logger.info(f"[SessionManager] 清理空闲会话: {sid}")


# 全局单例
session_manager = SessionManager(idle_timeout_sec=600)
