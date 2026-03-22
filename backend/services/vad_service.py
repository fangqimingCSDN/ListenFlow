"""
VAD (Voice Activity Detection) 服务
使用 Silero VAD 检测语音活动，实现智能断句。

断句策略（双重保障）：
  1. 静音时长断句：Silero VAD 检测到 end 事件
  2. 最大时长断句：单段超过 vad_max_segment_duration 秒强制截断

输入格式: 16kHz, float32, mono
Silero VAD 推荐窗口: 512 samples = 32ms @16kHz
"""
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
from loguru import logger
from ..core.config import settings


SAMPLE_RATE = 16000
CHUNK_SIZE = 512   # Silero VAD 固定窗口大小


class VADService:
    """
    Silero VAD 流式封装（每个 WebSocket 会话独立实例）。
    """

    def __init__(self):
        self._vad_model = None
        self._vad_iter = None
        self._load_model()

        # 流式状态
        self._buffer: np.ndarray = np.array([], dtype=np.float32)
        self._speech_buffer: np.ndarray = np.array([], dtype=np.float32)
        self._prev_chunks: deque = deque(maxlen=10)
        self._is_speaking: bool = False
        self._speech_start_time: float = 0.0
        self._total_samples: int = 0
        self._max_speech_samples = settings.vad_max_segment_duration * SAMPLE_RATE

    def _load_model(self):
        """加载 Silero VAD"""
        try:
            import torch
            logger.info("加载 Silero VAD 模型...")
            self._vad_model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self._vad_model.eval()
            from silero_vad import VADIterator
            self._vad_iter = VADIterator(
                self._vad_model,
                threshold=settings.vad_threshold,
                min_silence_duration_ms=settings.vad_silence_duration_ms,
            )
            logger.info("Silero VAD 加载完成")
        except Exception as e:
            logger.error(f"Silero VAD 加载失败: {e}")
            raise

    @staticmethod
    def bytes_to_float32(audio_bytes: bytes) -> np.ndarray:
        """16bit PCM bytes → float32 [-1, 1]"""
        arr = np.frombuffer(audio_bytes, dtype=np.int16)
        return arr.astype(np.float32) / 32768.0

    def process_chunk(
        self, audio_input
    ) -> List[Tuple[float, float, np.ndarray]]:
        """
        输入一个音频块，返回已完成的语音片段列表。
        每个片段: (start_sec, end_sec, audio_float32)
        """
        completed: List[Tuple[float, float, np.ndarray]] = []

        if isinstance(audio_input, bytes):
            audio_float = self.bytes_to_float32(audio_input)
        else:
            audio_float = audio_input.astype(np.float32)

        self._buffer = np.concatenate([self._buffer, audio_float])

        while len(self._buffer) >= CHUNK_SIZE:
            chunk = self._buffer[:CHUNK_SIZE]
            self._buffer = self._buffer[CHUNK_SIZE:]
            current_time = self._total_samples / SAMPLE_RATE
            self._total_samples += CHUNK_SIZE

            speech_dict = self._vad_iter(chunk, return_seconds=True)

            if speech_dict:
                if "start" in speech_dict and not self._is_speaking:
                    self._is_speaking = True
                    self._speech_start_time = current_time
                    prefix = (
                        np.concatenate(list(self._prev_chunks))
                        if self._prev_chunks
                        else np.array([], dtype=np.float32)
                    )
                    self._speech_buffer = np.concatenate([prefix, chunk])
                    logger.debug(f"[VAD] 语音开始 @ {current_time:.2f}s")

                if "end" in speech_dict and self._is_speaking:
                    self._speech_buffer = np.concatenate([self._speech_buffer, chunk])
                    seg = self._emit_segment(current_time)
                    if seg:
                        completed.append(seg)
                        logger.info(
                            f"[VAD] 静音断句: {seg[0]:.2f}s → {seg[1]:.2f}s "
                            f"({seg[1]-seg[0]:.2f}s)"
                        )
                    continue

            if self._is_speaking:
                self._speech_buffer = np.concatenate([self._speech_buffer, chunk])
                # 超过最大时长强制截断
                if len(self._speech_buffer) >= self._max_speech_samples:
                    seg = self._emit_segment(current_time)
                    if seg:
                        completed.append(seg)
                        logger.info(
                            f"[VAD] 最大时长强制断句: {seg[0]:.2f}s → {seg[1]:.2f}s"
                        )
            else:
                self._prev_chunks.append(chunk.copy())

        return completed

    def _emit_segment(
        self, end_time: float
    ) -> Optional[Tuple[float, float, np.ndarray]]:
        """输出当前语音片段并重置状态"""
        if len(self._speech_buffer) < CHUNK_SIZE:
            self._reset_speech_state()
            return None
        seg = (self._speech_start_time, end_time, self._speech_buffer.copy())
        self._reset_speech_state()
        return seg

    def _reset_speech_state(self):
        self._is_speaking = False
        self._speech_buffer = np.array([], dtype=np.float32)
        self._prev_chunks.clear()

    def flush(self) -> List[Tuple[float, float, np.ndarray]]:
        """流结束时强制输出剩余语音片段"""
        if self._is_speaking and len(self._speech_buffer) >= CHUNK_SIZE:
            end_time = self._total_samples / SAMPLE_RATE
            seg = self._emit_segment(end_time)
            if seg:
                return [seg]
        return []

    def reset(self):
        """重置所有状态（新会话）"""
        self._buffer = np.array([], dtype=np.float32)
        self._speech_buffer = np.array([], dtype=np.float32)
        self._prev_chunks.clear()
        self._is_speaking = False
        self._speech_start_time = 0.0
        self._total_samples = 0
        if self._vad_iter:
            self._vad_iter.reset_states()
        logger.debug("[VAD] 状态已重置")
