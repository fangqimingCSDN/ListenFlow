"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-03-21
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # sessions 表
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="recording"),
        sa.Column("audio_object_key", sa.String(512), nullable=True),
        sa.Column("audio_duration_sec", sa.Float(), nullable=True),
        sa.Column("audio_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("transcript_object_key", sa.String(512), nullable=True),
        sa.Column("language", sa.String(10), nullable=False, server_default="zh"),
        sa.Column("speaker_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
    )

    # speakers 表
    op.create_table(
        "speakers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("speaker_label", sa.String(50), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("embedding", postgresql.JSON(), nullable=True),
        sa.Column("color", sa.String(7), nullable=False, server_default="#4A90E2"),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"],
                                ondelete="CASCADE"),
    )
    op.create_index("ix_speakers_session_label", "speakers",
                    ["session_id", "speaker_label"], unique=True)

    # transcript_segments 表
    op.create_table(
        "transcript_segments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("speaker_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("is_final", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("language", sa.String(10), nullable=False, server_default="zh"),
        sa.Column("words", postgresql.JSON(), nullable=True),
        sa.Column("sequence_no", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=False,
                  server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"],
                                ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"],
                                ondelete="SET NULL"),
    )
    op.create_index("ix_segments_session_seq", "transcript_segments",
                    ["session_id", "sequence_no"])
    op.create_index("ix_segments_session_time", "transcript_segments",
                    ["session_id", "start_time"])


def downgrade() -> None:
    op.drop_table("transcript_segments")
    op.drop_table("speakers")
    op.drop_table("sessions")
