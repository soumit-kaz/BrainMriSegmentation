from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Integer, String, Text, create_engine, delete, func, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from inference_service.domain.segmentation_job import (
    ACTIVE_JOB_STATUSES,
    JobStatus,
    SegmentationJob,
    SegmentationNotFoundError,
    SegmentationSummary,
)
from inference_service.utils import get_utc_now

SCHEMA_LOCK_IDENTIFIER = 734_102_001


class DatabaseModel(DeclarativeBase):
    pass


class SegmentationJobRecord(DatabaseModel):
    __tablename__ = "segmentation_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    case_name: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(16), index=True)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    has_ground_truth: Mapped[bool] = mapped_column(Boolean)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    def to_domain(self) -> SegmentationJob:
        return SegmentationJob(
            job_id=self.job_id,
            case_name=self.case_name,
            status=JobStatus(self.status),
            input_hash=self.input_hash,
            has_ground_truth=self.has_ground_truth,
            attempt_count=self.attempt_count,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            error_message=self.error_message,
            summary=SegmentationSummary(**self.summary) if self.summary else None,
        )


class PostgresSegmentationJobRepository:
    def __init__(self, database_url: str, pool_size: int):
        self.engine = create_engine(database_url, pool_size=pool_size, pool_pre_ping=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(:lock_identifier)"), {"lock_identifier": SCHEMA_LOCK_IDENTIFIER})
            DatabaseModel.metadata.create_all(connection)

    def is_available(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError:
            return False

    @staticmethod
    def get_record(session: Session, job_id: str) -> SegmentationJobRecord:
        record = session.get(SegmentationJobRecord, job_id)
        if record is None:
            raise SegmentationNotFoundError(job_id)
        return record

    def add_job(self, job: SegmentationJob) -> None:
        with self.session_factory.begin() as session:
            session.add(SegmentationJobRecord(
                job_id=job.job_id, case_name=job.case_name, status=job.status, input_hash=job.input_hash,
                has_ground_truth=job.has_ground_truth, attempt_count=job.attempt_count, created_at=job.created_at,
                started_at=job.started_at, completed_at=job.completed_at, error_message=job.error_message, summary=None,
            ))

    def get_job(self, job_id: str) -> SegmentationJob:
        with self.session_factory() as session:
            return self.get_record(session, job_id).to_domain()

    def get_jobs(self, limit: int) -> list[SegmentationJob]:
        with self.session_factory() as session:
            records = session.scalars(select(SegmentationJobRecord).order_by(SegmentationJobRecord.created_at.desc()).limit(limit))
            return [record.to_domain() for record in records]

    def get_reusable_job(self, input_hash: str) -> SegmentationJob | None:
        with self.session_factory() as session:
            record = session.scalars(
                select(SegmentationJobRecord)
                .where(SegmentationJobRecord.input_hash == input_hash, SegmentationJobRecord.status != JobStatus.FAILED)
                .order_by(SegmentationJobRecord.created_at.desc())
                .limit(1)
            ).first()
            return record.to_domain() if record else None

    def get_job_count(self, statuses: frozenset[JobStatus]) -> int:
        with self.session_factory() as session:
            return session.scalar(
                select(func.count()).select_from(SegmentationJobRecord).where(SegmentationJobRecord.status.in_(list(statuses)))
            ) or 0

    def get_queue_position(self, job: SegmentationJob) -> int:
        if job.status != JobStatus.QUEUED:
            return 0
        with self.session_factory() as session:
            jobs_ahead = session.scalar(
                select(func.count()).select_from(SegmentationJobRecord).where(
                    SegmentationJobRecord.status.in_(list(ACTIVE_JOB_STATUSES)),
                    SegmentationJobRecord.created_at < job.created_at,
                )
            ) or 0
            return jobs_ahead + 1

    def mark_processing(self, job_id: str) -> SegmentationJob:
        with self.session_factory.begin() as session:
            record = self.get_record(session, job_id)
            record.status = JobStatus.PROCESSING
            record.started_at = get_utc_now()
            record.attempt_count += 1
            return record.to_domain()

    def mark_completed(self, job_id: str, summary: SegmentationSummary) -> SegmentationJob:
        with self.session_factory.begin() as session:
            record = self.get_record(session, job_id)
            record.status = JobStatus.COMPLETED
            record.completed_at = get_utc_now()
            record.error_message = None
            record.summary = asdict(summary)
            return record.to_domain()

    def mark_failed(self, job_id: str, error_message: str) -> SegmentationJob:
        with self.session_factory.begin() as session:
            record = self.get_record(session, job_id)
            record.status = JobStatus.FAILED
            record.completed_at = get_utc_now()
            record.error_message = error_message
            return record.to_domain()

    def delete_job(self, job_id: str) -> None:
        with self.session_factory.begin() as session:
            session.execute(delete(SegmentationJobRecord).where(SegmentationJobRecord.job_id == job_id))

    def get_expired_job_ids(self, completed_before: datetime) -> list[str]:
        with self.session_factory() as session:
            return list(session.scalars(
                select(SegmentationJobRecord.job_id).where(
                    SegmentationJobRecord.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                    SegmentationJobRecord.completed_at < completed_before,
                )
            ))
