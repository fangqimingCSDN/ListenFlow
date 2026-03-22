"""
说话人识别服务
使用 ERes2Net (iic/speech_eres2net_sv_zh-cn_16k-common) 提取声纹嵌入，
配合 OnlineSpeakerCluster 实现流式说话人聚类。

架构说明:
  - 声纹提取: ERes2Net，阿里达摩院开源，16kHz中文优化
  - 聚类算法: 余弦相似度 + EMA中心更新，O(n_speakers)复杂度
  - 每个WebSocket会话持有独立的聚类实例（状态隔离）

说话人标签格式: speaker_0, speaker_1, speaker_2 ...
用户可通过 REST API 将 speaker_0 映射为 "张某某"
"""
import numpy as np
from typing import List, Dict, Optional, Any
from loguru import logger


class OnlineSpeakerCluster:
    """
    流式在线说话人聚类（余弦相似度 + EMA更新）。

    Args:
        thr: 余弦相似度阈值，超过则归入已有说话人，否则新建
        ema: EMA更新系数，center = ema*new + (1-ema)*old
    """

    def __init__(self, thr: float = 0.55, ema: float = 0.9):
        self.thr = thr
        self.ema = ema
        self.centers: List[np.ndarray] = []

    def update(self, x: np.ndarray) -> int:
        """输入声纹嵌入，返回说话人ID（从0开始）"""
        if len(self.centers) == 0:
            self.centers.append(x.copy())
            return 0

        x_norm = x / (np.linalg.norm(x) + 1e-12)
        centers_norm = np.stack(
            [c / (np.linalg.norm(c) + 1e-12) for c in self.centers]
        )
        sims = centers_norm @ x_norm  # [n_speakers]
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.thr:
            # EMA 更新聚类中心
            self.centers[best_idx] = (
                self.ema * x + (1 - self.ema) * self.centers[best_idx]
            )
            return best_idx
        else:
            self.centers.append(x.copy())
            return len(self.centers) - 1

    def get_speaker_count(self) -> int:
        return len(self.centers)

    def reset(self):
        self.centers.clear()


class SpeakerService:
    """
    声纹提取服务（全局单例，模型只加载一次）。
    聚类实例由 SessionManager 按会话持有。
    """

    def __init__(self):
        self._pipeline = None
        self._load_model()

    def _load_model(self):
        """加载 ERes2Net 说话人验证模型"""
        try:
            logger.info("加载 ERes2Net 声纹提取模型...")
            from modelscope.pipelines import pipeline
            self._pipeline = pipeline(
                task="speaker-verification",
                model="iic/speech_eres2net_sv_zh-cn_16k-common",
            )
            logger.info("ERes2Net 声纹模型加载完成")
        except Exception as e:
            logger.error(f"声纹模型加载失败: {e}")
            raise

    def extract_embedding_sync(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> Optional[np.ndarray]:
        """
        同步提取声纹嵌入向量。在线程池中调用，不阻塞事件循环。

        Args:
            audio: float32 numpy 数组，值域[-1,1]
            sample_rate: 采样率，默认16000

        Returns:
            归一化声纹嵌入向量 (D,) 或 None
        """
        try:
            # ERes2Net 期望 int16 范围
            audio_in = audio * 32768.0
            result = self._pipeline([audio_in], output_emb=True)
            emb = np.array(result["embs"]).squeeze()
            # L2 归一化
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            return emb
        except Exception as e:
            logger.error(f"声纹提取失败: {e}")
            return None

    async def extract_embedding(
        self, audio: np.ndarray
    ) -> Optional[np.ndarray]:
        """异步声纹提取（派发到线程池）"""
        import asyncio
        return await asyncio.to_thread(self.extract_embedding_sync, audio)


# ── 全局单例 ─────────────────────────────────────────────────────────────────
_speaker_service: Optional[SpeakerService] = None


def get_speaker_service() -> SpeakerService:
    global _speaker_service
    if _speaker_service is None:
        _speaker_service = SpeakerService()
    return _speaker_service
