# -*- coding: utf-8 -*-
import pathlib

path = pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/backend/services/asr_service.py')
c = path.read_text(encoding='utf-8')

QWEN3_CLASS = '''
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
                f"[Qwen3-ASR] \'{text[:40]}\' | audio={end_time - start_time:.1f}s"
                f" | elapsed={elapsed:.2f}s | RTF={elapsed / (end_time - start_time + 1e-6):.2f}"
            )
            return ASRResult(text=text, language="zh", start_time=start_time, end_time=end_time)
        except Exception as e:
            logger.error(f"[Qwen3-ASR] transcribe failed: {e}")
            return None

'''

# Insert Qwen3 class before ASRService class
MARKER = '# ---------------------------------------------------------------------------\n# ASRService'
if MARKER not in c:
    # fallback marker
    MARKER = 'class ASRService:'
    insert_pos = c.find(MARKER)
else:
    insert_pos = c.find(MARKER)

c = c[:insert_pos] + QWEN3_CLASS + c[insert_pos:]

# Patch ASRService.__init__ to add qwen3 branch
OLD_INIT = '''        backend_name = getattr(settings, "asr_backend", "funasr").lower()
        if backend_name == "whisper":
            self._backend = WhisperBackend()
        else:
            self._backend = FunASRBackend()'''

NEW_INIT = '''        backend_name = getattr(settings, "asr_backend", "funasr").lower()
        if backend_name == "whisper":
            self._backend = WhisperBackend()
        elif backend_name == "qwen3":
            self._backend = Qwen3ASRBackend()
        else:
            self._backend = FunASRBackend()'''

if OLD_INIT in c:
    c = c.replace(OLD_INIT, NEW_INIT)
    print('ASRService init patched')
else:
    print('WARNING: ASRService init pattern not found, check manually')

path.write_text(c, encoding='utf-8')
print(f'DONE, total len: {len(c)}')
