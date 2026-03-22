import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, JSON, BigInteger, Index
)
from sqlalchemy.dialects.postgresql import UUID
from .database import Base


class Session(Base):
    """
    录音会话表。每次开始录音创建一条记录。

    与其他表的关系（逻辑关联，非数据库外键）:
      sessions.id  <--  speakers.session_id
      sessions.id  <--  transcript_segments.session_id

    为何不用外键 ForeignKey:
      1. 高并发写入时，外键约束会产生行级锁，降低吞吐量
      2. 未来分库分表时，跨库外键无法维护
      3. 通过代码逻辑（查询时用 WHERE session_id = :id）保证关联一致性
    """
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=True)
    status = Column(String(20), default="recording", nullable=False)
    audio_object_key = Column(String(512), nullable=True)
    audio_duration_sec = Column(Float, nullable=True)
    audio_size_bytes = Column(BigInteger, nullable=True)
    transcript_object_key = Column(String(512), nullable=True)
    language = Column(String(10), default="zh", nullable=False)
    speaker_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    # 无 ForeignKey，无 relationship
    # 查询关联: SELECT * FROM speakers WHERE session_id = :id


class Speaker(Base):
    """
    说话人表。存储说话人标签与真实姓名的映射。

    session_id: 逻辑关联 sessions.id（无数据库外键约束）
    speaker_label: 系统自动分配，如 speaker_0 / speaker_1
    display_name:  用户可编辑真实姓名，如 张某某
    """
    __tablename__ = "speakers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    speaker_label = Column(String(50), nullable=False)
    display_name = Column(String(100), nullable=True)
    embedding = Column(JSON, nullable=True)
    color = Column(String(7), default="#4A90E2", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        # 同一会话内 speaker_label 唯一（代码层保证，不靠外键）
        Index("ix_speakers_session_label", "session_id", "speaker_label", unique=True),
    )


class TranscriptSegment(Base):
    """
    转写文本片段表。每个 VAD 断句产生一条记录。

    session_id: 逻辑关联 sessions.id
    speaker_id: 逻辑关联 speakers.id（可为空，表示说话人未识别）
    """
    __tablename__ = "transcript_segments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), nullable=False)
    speaker_id = Column(UUID(as_uuid=True), nullable=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    confidence = Column(Float, default=1.0)
    is_final = Column(Boolean, default=True)
    language = Column(String(10), default="zh")
    words = Column(JSON, nullable=True)
    sequence_no = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_segments_session_seq",  "session_id", "sequence_no"),
        Index("ix_segments_session_time", "session_id", "start_time"),
        Index("ix_segments_speaker",      "speaker_id"),
    )
