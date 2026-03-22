import asyncio
from typing import Optional, List, Dict, Any
from loguru import logger
from ..core.config import settings

VECTOR_DIM = 384
COLLECTION_NAME = "speech_transcripts"


class TextEmbedder:
    """
    文本向量化，使用 sentence-transformers 多语言 MiniLM 模型。
    支持中文，384维向量，CPU 可快速推理（~5ms/句）。
    """

    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[Milvus] loading MiniLM-L12-v2 text embedder...")
            self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("[Milvus] text embedder loaded")
        except Exception as e:
            logger.error(f"[Milvus] text embedder load failed: {e}")
            raise

    def encode(self, text: str) -> List[float]:
        """text -> 384-dim float list"""
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    async def encode_async(self, text: str) -> List[float]:
        """async encode (thread pool)"""
        return await asyncio.to_thread(self.encode, text)


class MilvusService:
    """
    Milvus 向量库服务。

    启用方式: .env 设置 ENABLE_VECTOR_STORE=true VECTOR_BACKEND=milvus

    Collection Schema:
      id             INT64 PK auto
      session_id     VARCHAR(64)
      speaker_label  VARCHAR(32)
      speaker_display VARCHAR(64)
      start_time     FLOAT
      end_time       FLOAT
      text           VARCHAR(2000)
      embedding      FLOAT_VECTOR(384)  <- MiniLM 输出
    """

    def __init__(self):
        self._col = None
        self._embedder = None
        self._connect()

    def _connect(self):
        try:
            from pymilvus import connections, Collection, utility
            host = getattr(settings, "milvus_host", "localhost")
            port = getattr(settings, "milvus_port", 19530)
            logger.info(f"[Milvus] connecting to {host}:{port}")
            connections.connect("default", host=host, port=port)

            if not utility.has_collection(COLLECTION_NAME):
                self._create_collection()
            else:
                logger.info(f"[Milvus] collection '{COLLECTION_NAME}' exists")

            self._col = Collection(COLLECTION_NAME)
            self._col.load()
            self._embedder = TextEmbedder()
            logger.info("[Milvus] ready")
        except Exception as e:
            logger.error(f"[Milvus] connect failed: {e}")
            raise

    def _create_collection(self):
        """创建集合 + HNSW 向量索引 + session_id 标量索引"""
        from pymilvus import FieldSchema, CollectionSchema, DataType, Collection
        fields = [
            FieldSchema(name="id",               dtype=DataType.INT64,        is_primary=True, auto_id=True),
            FieldSchema(name="session_id",       dtype=DataType.VARCHAR,      max_length=64),
            FieldSchema(name="speaker_label",    dtype=DataType.VARCHAR,      max_length=32),
            FieldSchema(name="speaker_display",  dtype=DataType.VARCHAR,      max_length=64),
            FieldSchema(name="start_time",       dtype=DataType.FLOAT),
            FieldSchema(name="end_time",         dtype=DataType.FLOAT),
            FieldSchema(name="text",             dtype=DataType.VARCHAR,      max_length=2000),
            FieldSchema(name="embedding",        dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM),
        ]
        schema = CollectionSchema(fields, description="Speech transcripts semantic vectors")
        col = Collection(COLLECTION_NAME, schema)
        col.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
        col.create_index(field_name="session_id")
        logger.info(f"[Milvus] collection '{COLLECTION_NAME}' created with HNSW/COSINE index")

    # ── 写入 ─────────────────────────────────────────────────────────────────

    def insert_segment_sync(
        self,
        session_id: str,
        speaker_label: str,
        speaker_display: str,
        start_time: float,
        end_time: float,
        text: str,
    ) -> Optional[int]:
        """同步写入一条转写片段（在线程池中调用）"""
        try:
            vec = self._embedder.encode(text)
            result = self._col.insert([
                [session_id[:64]],
                [speaker_label[:32]],
                [speaker_display[:64]],
                [float(start_time)],
                [float(end_time)],
                [text[:2000]],
                [vec],
            ])
            self._col.flush()
            inserted_id = result.primary_keys[0]
            logger.debug(f"[Milvus] inserted id={inserted_id}: '{text[:30]}'")
            return inserted_id
        except Exception as e:
            logger.error(f"[Milvus] insert failed: {e}")
            return None

    async def insert_segment(
        self,
        session_id: str,
        speaker_label: str,
        speaker_display: str,
        start_time: float,
        end_time: float,
        text: str,
    ) -> Optional[int]:
        """异步写入（fire-and-forget 友好，派发到线程池）"""
        return await asyncio.to_thread(
            self.insert_segment_sync,
            session_id, speaker_label, speaker_display,
            start_time, end_time, text,
        )

    # ── 语义搜索 ─────────────────────────────────────────────────────────────

    def search_sync(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        speaker_label: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        按语义搜索历史转写片段。
        支持按 session_id / speaker_label 过滤。
        """
        try:
            query_vec = self._embedder.encode(query_text)
            filters = []
            if session_id:
                filters.append(f'session_id == "{session_id}"')
            if speaker_label:
                filters.append(f'speaker_label == "{speaker_label}"')
            expr = " && ".join(filters) if filters else ""

            results = self._col.search(
                data=[query_vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k,
                expr=expr or None,
                output_fields=[
                    "session_id", "speaker_label", "speaker_display",
                    "start_time", "end_time", "text",
                ],
            )
            hits = []
            for hit in results[0]:
                hits.append({
                    "id":              hit.id,
                    "score":           round(hit.score, 4),
                    "text":            hit.entity.get("text"),
                    "session_id":      hit.entity.get("session_id"),
                    "speaker_label":   hit.entity.get("speaker_label"),
                    "speaker_display": hit.entity.get("speaker_display"),
                    "start_time":      hit.entity.get("start_time"),
                    "end_time":        hit.entity.get("end_time"),
                })
            return hits
        except Exception as e:
            logger.error(f"[Milvus] search failed: {e}")
            return []

    async def search(
        self,
        query_text: str,
        session_id: Optional[str] = None,
        speaker_label: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """异步语义搜索"""
        return await asyncio.to_thread(
            self.search_sync, query_text, session_id, speaker_label, top_k
        )

    def delete_session_sync(self, session_id: str):
        """删除某会话的所有向量"""
        try:
            self._col.delete(expr=f'session_id == "{session_id}"')
            self._col.flush()
            logger.info(f"[Milvus] deleted vectors for session {session_id}")
        except Exception as e:
            logger.error(f"[Milvus] delete failed: {e}")

    async def delete_session(self, session_id: str):
        """异步删除"""
        await asyncio.to_thread(self.delete_session_sync, session_id)


# ── 全局单例 ─────────────────────────────────────────────────────────────────
_milvus_service: Optional[MilvusService] = None


def get_milvus_service() -> Optional[MilvusService]:
    """
    获取 Milvus 服务实例。
    未启用（ENABLE_VECTOR_STORE=false）时返回 None，调用方无需判断是否配置。
    """
    global _milvus_service
    if not getattr(settings, "enable_vector_store", False):
        return None
    if _milvus_service is None:
        _milvus_service = MilvusService()
    return _milvus_service
