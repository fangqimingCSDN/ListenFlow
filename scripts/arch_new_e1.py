# -*- coding: utf-8 -*-
import pathlib

SEC6A = '''## 七、模型选型决策

> **选型方法论**：每个组件从五个维度评估：
> 📊 准确率 / ⚡ 延迟 / 💾 资源占用 / 🔧 部署难度 / 🌊 流式兼容性

---

### 7.1 语音识别（ASR）选型

#### 7.1.1 全市场方案横向对比

| 方案 | 开发方 | 架构类型 | 中文 CER | RTF(GPU) | 显存 | 多语言 | 词级时间戳 | 国内部署 |
|------|--------|---------|---------|---------|------|-------|----------|----------|
| ✅ **paraformer-zh** | 达摩院 | 非自回归 CIF | ~3% | 0.05~0.1 | 2 GB | 中文为主 | ❌ | ✅ 免 Token |
| ⭐ **Qwen3-ASR-1.7B** | 阿里云 | LLM Decoder-Only | ~2% | 0.15~0.3 | 4 GB | 50+ 种 | ✅ | ✅ 免 Token |
| SenseVoice-S | 达摩院 | 非自回归 | ~2.5% | 0.03~0.08 | 1.5 GB | 50+ 种 | ✅ | ✅ 免 Token |
| faster-whisper v3 | OpenAI | Encoder-Decoder | ~5% | 0.2~0.4 | 6 GB | 99 种 | ✅ | ✅ 免 Token |
| Whisper medium | OpenAI | Encoder-Decoder | ~7% | 0.1~0.2 | 2.5 GB | 99 种 | ✅ | ✅ 免 Token |
| wav2vec2 | Meta | CTC | ~6% | 快 | 1 GB | 极少 | ❌ | ⚠️ 需申请 |

#### 7.1.2 三大架构本质差异

```
① 非自回归（paraformer）速度最快
   音频 → 一次前向传播 → [你][好][世][界] 同时输出
   RTF ≈ 0.05，速度是 Whisper 的 5~10x

② 自回归 Encoder-Decoder（Whisper）均衡
   音频 → Encoder → Decoder 逐 token 生成
   生成 4 字 = 4 次前向传播  RTF ≈ 0.3
   优势：词级时间戳，99 种语言

③ LLM Decoder-Only（Qwen3-ASR）精度最高
   音频 → 动态编码器 → LLM 自回归生成
   支持 system prompt，无音频长度限制
   中文 CER 最低（~2%），专业术语识别强
```

#### 7.1.3 选型决策

| 后端 | 场景 | 切换方式 |
|------|------|----------|
| ✅ **funasr**（默认）| 中文会议，速度优先 | ASR_BACKEND=funasr |
| faster-whisper | 多语言/词级时间戳 | ASR_BACKEND=whisper |
| Qwen3-ASR | 中文最高精度/专业术语 | ASR_BACKEND=qwen3 |

**未采用**：SenseVoice（与 paraformer 重叠）/ wav2vec2（中文弱无标点）/ Conformer（精度低于离线模式）

'''

pathlib.Path('d:/B-Work/PyCharm/2025/speech_proj/scripts/_arch_sec6a.txt').write_text(SEC6A, encoding='utf-8')
print('sec6a len:', len(SEC6A))
print('done')
