# -*- coding: utf-8 -*-
import pathlib

f = pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
c = f.read_text(encoding='utf-8')

OLD2 = '### 7.2 VAD \u9009\u578b'
NEW2 = '### 7.2 VAD \u9009\u578b\n\n#### 7.2.1 Market Overview\n\n| Solution | Arch | Window | Accuracy | CPU | Streaming | Deploy |\n|------|------|------|------|------|------|------|\n| **Silero VAD** | LSTM | 32ms | High | ~1ms/frame | Yes | pip |\n| WebRTC VAD | Signal | 10ms | Medium | Very low | Yes | C binding |\n| pyannote VAD | Transformer | frame | Highest | needs GPU | No | HF token req |\n| FSMN-VAD | FSMN | 10ms | High | Medium | Yes | bundled FunASR |\n| Energy-based | threshold | any | Low | Minimal | Yes | no deps |\n\n#### 7.2.2 Decision\n\n**Selected: Silero VAD**\n- High accuracy, robust against classroom noise and background music\n- 32ms window matches project CHUNK_SIZE=512 design\n- pip install, no token required\n- Three key parameters configurable via .env\n\n**Not selected: WebRTC VAD** - energy-based algorithm, high miss rate for distant/soft speech.\n\n**Not selected: pyannote VAD** - needs GPU + HF token, no streaming input support.\n\n**Not selected: FSMN-VAD** - bundled in FunASR but needs independent per-session VAD instance; silence regression risk.\n\nDual-trigger segmentation strategy:\n- Condition A: silence > `VAD_SILENCE_DURATION_MS` (default 800ms)\n- Condition B: segment length > `VAD_MAX_SEGMENT_DURATION` (default 30s) force-cut\n'
c = c.replace(OLD2, NEW2, 1)
print('patch2:', 'ok' if '7.2.1' in c else 'FAIL')

f.write_text(c, encoding='utf-8')
print('saved len:', len(c))
