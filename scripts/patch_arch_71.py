# -*- coding: utf-8 -*-
import pathlib

f = pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/ARCHITECTURE.md')
c = f.read_text(encoding='utf-8')

# ── patch 1: add section 7.0 preamble ──────────────────────────────────────
OLD0 = '## \u4e03\u3001\u6a21\u578b\u9009\u578b\u51b3\u7b56'
NEW0 = '''## \u4e03\u3001\u6a21\u578b\u9009\u578b\u51b3\u7b56

> \u6bcf\u4e2a\u7ec4\u4ef6\u5747\u4ece\u4ee5\u4e0b\u4e94\u4e2a\u7ef4\u5ea6\u8bc4\u4f30\uff1a\u51c6\u786e\u7387 / \u5ef6\u8fdf / \u8d44\u6e90\u5360\u7528 / \u90e8\u7f72\u96be\u5ea6 / \u6d41\u5f0f\u5e73\u514c\u6027
'''
c = c.replace(OLD0, NEW0, 1)
print('patch0:', 'ok' if NEW0 in c else 'FAIL')

# ── patch 2: replace section 7.1 ───────────────────────────────────────────
OLD1 = '### 7.1 ASR \u9009\u578b'
NEW1 = '''### 7.1 ASR \u9009\u578b

#### 7.1.1 \u5e02\u573a\u4e3b\u6d41\u65b9\u6848\u5168\u666f

| \u65b9\u6848 | \u5f00\u53d1\u65b9 | \u67b6\u6784 | \u4e2d\u6587 CER | RTF (GPU) | \u663e\u5b58 | \u591a\u8bed\u8a00 | \u8bcd\u7ea7\u65f6\u95f4\u6233 | HF Token |
|------|------|------|------|------|------|------|------|------|
| **paraformer-zh** | \u9759\u97f3\u79d1\u6280 | \u975e\u81ea\u56de\u5f52 | ~3% | 0.05~0.1 | 2 GB | \u4e2d\u6587\u4e3a\u4e3b | \u4e0d\u652f\u6301 | \u4e0d\u9700 |
| **Qwen3-ASR-1.7B** | \u963f\u91cc | LLM decoder-only | ~2% | 0.15~0.3 | 4 GB | 50+ | \u652f\u6301 | \u4e0d\u9700 |
| SenseVoice-S | \u9759\u97f3\u79d1\u6280 | \u975e\u81ea\u56de\u5f52 | ~2.5% | 0.03~0.08 | 1.5 GB | 50+ | \u652f\u6301 | \u4e0d\u9700 |
| faster-whisper v3 | OpenAI | Encoder-Decoder | ~5% | 0.2~0.4 | 6 GB | 99 \u79cd | \u652f\u6301 | \u4e0d\u9700 |
| whisper medium | OpenAI | Encoder-Decoder | ~7% | 0.1~0.2 | 2.5 GB | 99 \u79cd | \u652f\u6301 | \u4e0d\u9700 |
| wav2vec2 | Meta | CTC | ~6% | \u5feb | 1 GB | \u5c11 | \u4e0d\u652f\u6301 | \u9700\u8981 |
| conformer | \u5404\u5bb6 | \u6d41\u5f0f CTC | ~4% | \u5f88\u5feb | 1.5 GB | \u6709\u9650 | \u4e0d\u652f\u6301 | \u5206\u60c5\u51b5 |

#### 7.1.2 \u6838\u5fc3\u67b6\u6784\u5dee\u5f02

```
\u81ea\u56de\u5f52 (Whisper):
  \u97f3\u9891 -> Encoder -> \u9010token\u751f\u6210 -> "\u4f60" "\u597d" "\u4e16" "\u754c"
  \u751f\u62104\u5b57 = 4\u6b21\u524d\u5411\u4f20\u64ad  <- \u6162

\u975e\u81ea\u56de\u5f52 (paraformer / SenseVoice):
  \u97f3\u9891 -> \u4e00\u6b21\u524d\u5411\u4f20\u64ad -> ["\u4f60" "\u597d" "\u4e16" "\u754c"] \u540c\u65f6\u8f93\u51fa
  \u751f\u62104\u5b57 = 1\u6b21\u524d\u5411\u4f20\u64ad  <- \u5feb 5~10x

LLM decoder-only (Qwen3-ASR):
  \u97f3\u9891 -> \u52a8\u6001\u957f\u5ea6\u7f16\u7801 -> LLM\u9010token\u751f\u6210
  \u652f\u6301 system prompt / \u65e0\u957f\u5ea6\u9650\u5236 / \u5f3a\u5927\u4e0a\u4e0b\u6587
  \u901f\u5ea6\u4ecb\u4e8e Whisper \u548c paraformer \u4e4b\u95f4
```

#### 7.1.3 \u9009\u578b\u51b3\u7b56

**\u9ed8\u8ba4\u9009 paraformer-zh (FunASR)**
- \u4e2d\u6587\u573a\u666f\u901f\u5ea6\u6700\u5feb\uff0cRTF 0.05\uff0c\u5b9e\u65f6\u7387\u8fdc\u5927\u4e8e 1
- \u4e0d\u9700 HF token\uff0c\u56fd\u5185\u76f4\u63a5 modelscope \u4e0b\u8f7d
- \u4ec5 2 GB \u663e\u5b58\uff0c\u6e38\u620f\u672c / \u5de5\u4f5c\u7ad9 GPU \u53ef\u8fd0\u884c
- \u5185\u7f6e VAD + \u6807\u70b9\u6062\u590d\uff0c\u8f93\u51fa\u8d28\u91cf\u9ad8

**\u53ef\u5207\u6362 faster-whisper large-v3**\uff0c\u9002\u7528\u573a\u666f\uff1a
- \u9700\u8981\u8bcd\u7ea7\u65f6\u95f4\u6233
- \u591a\u8bed\u8a00 / \u8bed\u79cd\u4e0d\u786e\u5b9a\u7684\u6df7\u5408\u5f55\u97f3
- \u6b63\u786e\u7387\u8981\u6c42\u9ad8\u4e8e\u901f\u5ea6\n
**\u53ef\u5207\u6362 Qwen3-ASR-1.7B**\uff0c\u9002\u7528\u573a\u666f\uff1a
- \u4e2d\u6587 CER \u8981\u6c42\u6700\u9ad8\uff08\u8d85\u8fc7 paraformer\uff09
- \u9700\u8981 system prompt \u63a7\u5236\u8f93\u51fa\u98ce\u683c\uff08\u5982\u533b\u7597/\u6cd5\u5f8b\u672f\u8bed\uff09
- \u5bf9\u5ef6\u8fdf\u8981\u6c42\u4e0d\u5982 paraformer \u9ad8

**\u672a\u91c7\u7528 SenseVoice** \u539f\u56e0\uff1a\u4e0e paraformer \u540c\u5c5e\u9759\u97f3\u79d1\u6280\u751f\u6001\uff0c\u4e14\u672c\u9879\u76ee\u5df2\u901a\u8fc7 FunASR \u96c6\u6210 paraformer\uff0c\u5207\u6362\u6210\u672c\u9ad8\u3002

**\u672a\u91c7\u7528 wav2vec2** \u539f\u56e0\uff1a\u4e2d\u6587\u652f\u6301\u5f31\uff0c\u65e0\u6807\u70b9\uff0c\u4e0d\u9002\u5408\u4f1a\u8bae\u573a\u666f\u3002

\u5207\u6362\u65b9\u5f0f: .env \u8bbe\u7f6e `ASR_BACKEND=funasr` / `whisper` / `qwen3`\uff0c\u65e0\u9700\u6539\u4ee3\u7801
'''
c = c.replace(OLD1, NEW1, 1)
print('patch1:', 'ok' if '7.1.1' in c else 'FAIL')

f.write_text(c, encoding='utf-8')
print('saved, len:', len(c))
