"""
ASR (自动语音识别) 服务

支持两种后端（通过 .env 配置 ASR_BACKEND 切换）:
  funasr   : FunASR paraformer-zh  — 中文SOTA，速度极快，无需HuggingFace token
  whisper  : faster-whisper large-v3 — 多语言强，精度高，需要更大显存

线程安全说明:
  两种后端的 generate/transcribe 都是同步阻塞调用（占用 CPU/GPU 数秒）。
  通过 asyncio.to_thread() 派发到线程池执行，让 asyncio 事件循环
  在转写期间可以继续处理其他 WebSocket 消息（接收新音频、响应 pause/stop 等）。
  同时用 asyncio.Semaphore 限制最大并发转写数，防止 GPU 内存溢出。
"""
import time
from typing import Optional, List, Dict, Any
import numpy as np
from loguru import logger
from ..core.config import settings

# 最大并发 ASR 转写数（防止多段同时推理耗尽显存）
# GPU: 建议 1~2；CPU: 可适当调大
_ASR_SEMAPHORE: Optional["asyncio.Semaphore"] = None


def _get_semaphore():
    """延迟初始化信号量（需在 asyncio 事件循环内调用）"""
    import asyncio
    global _ASR_SEMAPHORE
    if _ASR_SEMAPHORE is None:
        max_concurrent = getattr(settings, "asr_max_concurrent", 2)
        _ASR_SEMAPHORE = asyncio.Semaphore(max_concurrent)
    return _ASR_SEMAPHORE


class ASRResult:
    """ASR 识别结果"""
    def __init__(
        self,
        text: str,
        language: str = "zh",
        start_time: float = 0.0,
        end_time: float = 0.0,
        confidence: float = 1.0,
        words: Optional[List[Dict]] = None,
    ):
        self.text = text
        self.language = language
        self.start_time = start_time
        self.end_time = end_time
        self.confidence = confidence
        self.words = words or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "words": self.words,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FunASR 后端
# ─────────────────────────────────────────────────────────────────────────────

class FunASRBackend:
    """
    FunASR paraformer-zh 后端。
    中文最优，速度是 Whisper 的 5-10x，无需 HuggingFace token，可完全离线。
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
        try:
            import torch
            from funasr import AutoModel
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            logger.info(f"[ASR] 加载 FunASR paraformer-zh @ {device}")
            self._model = AutoModel(
                model="paraformer-zh",
                model_revision="v2.0.4",
                vad_model="fsmn-vad",
                vad_model_revision="v2.0.4",
                punc_model="ct-punc-c",
                punc_model_revision="v2.0.4",
                device=device,
                disable_log=True,
            )
            logger.info("[ASR] FunASR 加载完成")
        except Exception as e:
            logger.error(f"[ASR] FunASR 加载失败: {e}")
            raise

    def transcribe_sync(
        self, audio: np.ndarray, start_time: float = 0.0, end_time: float = 0.0
    ) -> Optional[ASRResult]:
        if len(audio) == 0:
            return None
        try:
            t0 = time.perf_counter()
            result = self._model.generate(
                input=audio * 32768.0,
                batch_size_s=1,  
                language="zh",
            )
            elapsed = time.perf_counter() - t0
            if not result or not result[0].get("text"):
                return None
            # Import path varies across funasr versions - try all known locations
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
            except ImportError:
                try:
                    from funasr.utils.postprocess import rich_transcription_postprocess
                except ImportError:
                    import re as _re
                    def rich_transcription_postprocess(t):
                        t = _re.sub(r"\u003c[^\u003e]+\u003e", "", t)
                        return _re.sub(r"\s+", " ", t).strip()
            text = rich_transcription_postprocess(result[0]["text"]).strip()
            if not text:
                return None
            logger.debug(
                f"[FunASR] '{text[:40]}' | 音频={end_time-start_time:.1f}s "
                f"| 耗时={elapsed:.2f}s | RTF={elapsed/(end_time-start_time+1e-6):.2f}"
            )
            return ASRResult(text=text, language="zh",
                             start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.error(f"[FunASR] 转写失败: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# faster-whisper 后端
# ─────────────────────────────────────────────────────────────────────────────

class WhisperBackend:
    """
    faster-whisper large-v3 后端。
    多语言支持强，精度高，比原版 Whisper 快 4x（CTranslate2 后端）。
    GPU 推荐 float16；纯 CPU 推荐 int8（内存占用约 1.5GB）。

    对比 FunASR:
      优点: 多语言、自动语言检测、词级时间戳
      缺点: 中文速度略慢于 paraformer，需更多显存，自回归逐token生成
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
        try:
            import torch
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # GPU 用 float16；CPU 用 int8 节省内存
            compute_type = "float16" if device == "cuda" else "int8"
            model_size = getattr(settings, "whisper_model_size", "large-v3")
            model_path = getattr(settings, "whisper_model_path", "") or model_size
            logger.info(
                f"[ASR] 加载 faster-whisper {model_size} @ {device} / {compute_type}"
            )
            self._model = WhisperModel(
                model_path,
                device=device,
                compute_type=compute_type,
                num_workers=1,
                download_root=None,
            )
            logger.info("[ASR] faster-whisper 加载完成")
        except Exception as e:
            logger.error(f"[ASR] faster-whisper 加载失败: {e}")
            raise

    def transcribe_sync(
        self, audio: np.ndarray, start_time: float = 0.0, end_time: float = 0.0
    ) -> Optional[ASRResult]:
        if len(audio) == 0:
            return None
        try:
            t0 = time.perf_counter()
            # faster-whisper 期望 float32 [-1,1]
            segments_gen, info = self._model.transcribe(
                audio,
                language="zh",           # 指定语言跳过语言检测，更快
                beam_size=5,
                vad_filter=False,        # VAD 已在外层做，这里关闭避免重复
                word_timestamps=True,    # 词级时间戳
                condition_on_previous_text=False,  # 流式场景关闭，防止幻觉
            )
            # 消费生成器
            words = []
            text_parts = []
            for seg in segments_gen:
                text_parts.append(seg.text)
                if seg.words:
                    words.extend([
                        {"word": w.word, "start": w.start, "end": w.end,
                         "probability": w.probability}
                        for w in seg.words
                    ])
            elapsed = time.perf_counter() - t0
            text = "".join(text_parts).strip()
            if not text:
                return None
            logger.debug(
                f"[Whisper] '{text[:40]}' | 音频={end_time-start_time:.1f}s "
                f"| 耗时={elapsed:.2f}s | RTF={elapsed/(end_time-start_time+1e-6):.2f}"
            )
            return ASRResult(
                text=text,
                language=info.language,
                start_time=start_time,
                end_time=end_time,
                confidence=float(info.language_probability),
                words=words,
            )
        except Exception as e:
            logger.error(f"[Whisper] 转写失败: {e}")
            return None


# ─────────────────────────────────────────────────────────────────────────────
# ASRService: 统一接口 + 异步调度
# ─────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Qwen3-ASR backend
# ---------------------------------------------------------------------------

class Qwen3ASRBackend:
    """
    Qwen3-ASR-1.7B backend (offline, transformers + modelscope).

    Advantages vs paraformer/whisper:
      - Best Chinese CER: AISHELL ~2%, WenetSpeech ~3.5%
      - Supports system prompt to control output style / terminology
      - Native unlimited-length audio (no chunking needed)
      - Excellent noise robustness
      - 50+ languages

    Tradeoffs:
      - Autoregressive LLM inference, RTF ~0.15~0.3 (slower than paraformer)
      - Requires ~4GB VRAM
      - No official streaming API (mitigated by VAD chunking in this project)

    Deployment: offline model loaded via transformers + modelscope snapshot_download.
    Preferred over vLLM (low concurrency ASR, PagedAttention gain minimal)
    and over Docker-only (just packaging, does not improve inference).
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._load()

    def _load(self):
        try:
            import torch
            from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_path = getattr(settings, "qwen3_asr_model_path", "") or "Qwen/Qwen3-ASR-1.7B"

            # try modelscope snapshot_download for Chinese mirror
            if not pathlib.Path(model_path).exists():
                try:
                    from modelscope import snapshot_download
                    model_id = getattr(settings, "qwen3_asr_model_id", "Qwen/Qwen3-ASR-1.7B")
                    logger.info(f"[ASR] downloading Qwen3-ASR from ModelScope: {model_id}")
                    model_path = snapshot_download(model_id)
                except Exception:
                    pass  # fallback: load directly from HF hub by model id

            logger.info(f"[ASR] loading Qwen3-ASR @ {self._device}, path={model_path}")
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=self._device,
                trust_remote_code=True,
            )
            self._model.eval()
            logger.info("[ASR] Qwen3-ASR loaded")
        except Exception as e:
            logger.error(f"[ASR] Qwen3-ASR load failed: {e}")
            raise

    def transcribe_sync(
        self, audio: np.ndarray, start_time: float = 0.0, end_time: float = 0.0
    ) -> Optional[ASRResult]:
        if len(audio) == 0:
            return None
        try:
            import torch
            t0 = time.perf_counter()

            system_prompt = getattr(
                settings,
                "qwen3_asr_system_prompt",
                "Transcribe the speech into Simplified Chinese text with punctuation.",
            )
            conversation = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [{"type": "audio", "audio_url": audio}],
                },
            ]
            text_input = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self._processor(
                text=text_input,
                audios=[audio],
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1]
            text = self._processor.batch_decode(
                output_ids[:, input_len:], skip_special_tokens=True
            )[0].strip()

            elapsed = time.perf_counter() - t0
            if not text:
                return None
            logger.debug(
                f"[Qwen3-ASR] '{text[:40]}' | audio={end_time - start_time:.1f}s"
                f" | elapsed={elapsed:.2f}s | RTF={elapsed / (end_time - start_time + 1e-6):.2f}"
            )
            return ASRResult(text=text, language="zh", start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.error(f"[Qwen3-ASR] transcribe failed: {e}")
            return None

class ASRService:
    """
    统一 ASR 接口。根据 settings.asr_backend 选择后端。

    【异步工作原理】
    ┌─────────────────────────────────────────────────────────────┐
    │  asyncio 事件循环 (单线程)                                   │
    │                                                              │
    │  WebSocket接收循环                                           │
    │    ├─ 收到音频块 → VAD → 检测到语音结束                     │
    │    ├─ asyncio.create_task(_process_segment(...))  ← 立即返回 │
    │    └─ 继续接收下一个音频块 (不等转写完成)                   │
    │                                                              │
    │  _process_segment 任务:                                      │
    │    └─ await asr.transcribe(audio)                           │
    │         └─ await asyncio.to_thread(transcribe_sync, audio)  │
    │              └─ [线程池执行, 2~5秒] ← 不阻塞事件循环        │
    │                  转写完成后 → 推送 WebSocket transcript      │
    └─────────────────────────────────────────────────────────────┘

    【Semaphore 并发控制】
    同时最多 asr_max_concurrent(默认2) 个片段并行转写。
    超出的片段会等待信号量释放后再进入线程池。
    防止多段同时推理打爆 GPU 显存。
    """

    def __init__(self):
        backend_name = getattr(settings, "asr_backend", "funasr").lower()
        if backend_name == "whisper":
            self._backend = WhisperBackend()
        elif backend_name == "qwen3":
            self._backend = Qwen3ASRBackend()
        else:
            self._backend = FunASRBackend()
        logger.info(f"[ASRService] 使用后端: {backend_name}")

    async def transcribe(
        self,
        audio: np.ndarray,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> Optional[ASRResult]:
        """
        异步转写。

        执行流程:
          1. 获取信号量（最多 asr_max_concurrent 个同时转写）
          2. 通过 asyncio.to_thread 在线程池中运行同步转写
          3. 转写完成后释放信号量，推送结果

        不阻塞 WebSocket 主循环：转写在线程池跑，
        asyncio 事件循环仍然可以接收新音频包。
        """
        import asyncio
        sem = _get_semaphore()
        async with sem:   # 并发限流：超过上限的片段在这里等待
            return await asyncio.to_thread(
                self._backend.transcribe_sync, audio, start_time, end_time
            )


# ── 全局单例 ─────────────────────────────────────────────────────────────────
_asr_service: Optional[ASRService] = None


def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = ASRService()
    return _asr_service

