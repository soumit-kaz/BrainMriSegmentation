from __future__ import annotations

import logging
import multiprocessing
import queue
from multiprocessing.context import SpawnProcess
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from typing import Any

from segmentation_core.runtime import set_random_seed
from segmentation_core.settings import SegmentationSettings

from training_service.application.model_trainer import ModelTrainer
from training_service.domain.training_run import (
    TrainingCommand,
    TrainingEventType,
    TrainingRunRequest,
    TrainingState,
    TrainingStoppedError,
)
from training_service.helpers import configure_torch_performance, configure_training_logging
from training_service.infrastructure.training_history_repository import JsonTrainingHistoryRepository
from training_service.utils import get_utc_timestamp

logger = logging.getLogger("segmentation.training")

TrainingEventQueue = Queue[dict[str, Any]]


class QueueProgressReporter:
    def __init__(self, event_queue: TrainingEventQueue, stop_event: Event):
        self.event_queue = event_queue
        self.stop_event = stop_event

    def report(self, event: dict[str, Any]) -> None:
        self.event_queue.put({**event, "timestamp": get_utc_timestamp()})

    def is_stop_requested(self) -> bool:
        return self.stop_event.is_set()


class QueueLogHandler(logging.Handler):
    def __init__(self, progress_reporter: QueueProgressReporter):
        super().__init__()
        self.progress_reporter = progress_reporter

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.progress_reporter.report({"type": TrainingEventType.LOG, "level": record.levelname, "message": record.getMessage()})
        except Exception:
            self.handleError(record)


def run_training_job(settings: SegmentationSettings, run_request: TrainingRunRequest, progress_reporter: QueueProgressReporter) -> None:
    set_random_seed(settings.random_seed)
    configure_torch_performance()
    try:
        model_trainer = ModelTrainer(settings, progress_reporter, JsonTrainingHistoryRepository(settings))
        if run_request.command == TrainingCommand.TRAIN:
            model_trainer.run_training(run_request.is_resume_requested)
        else:
            model_trainer.run_test_evaluation()
        finished_state, finished_message = TrainingState.COMPLETED, "Run completed."
    except TrainingStoppedError:
        logger.info("Run stopped by request. It can be resumed from the last completed epoch.")
        finished_state, finished_message = TrainingState.STOPPED, "Stopped by request."
    except Exception as error:
        logger.exception("Run failed.")
        finished_state, finished_message = TrainingState.FAILED, str(error)
    progress_reporter.report({"type": TrainingEventType.RUN_FINISHED, "state": finished_state, "message": finished_message})


def run_training_process(settings: SegmentationSettings, run_request: TrainingRunRequest,
                         event_queue: TrainingEventQueue, stop_event: Event) -> None:
    progress_reporter = QueueProgressReporter(event_queue, stop_event)
    configure_training_logging(settings, QueueLogHandler(progress_reporter))
    run_training_job(settings, run_request, progress_reporter)


class SpawnedTrainingProcessHandle:
    def __init__(self, process: SpawnProcess, event_queue: TrainingEventQueue, stop_event: Event):
        self.process = process
        self.event_queue = event_queue
        self.stop_event = stop_event

    def get_event(self, timeout_seconds: float) -> dict[str, Any] | None:
        try:
            return self.event_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def get_exit_code(self) -> int | None:
        return self.process.exitcode

    def request_stop(self) -> None:
        self.stop_event.set()

    def stop(self, grace_seconds: float) -> None:
        self.stop_event.set()
        self.process.join(timeout=grace_seconds)
        if self.process.is_alive():
            self.process.terminate()


class SpawnedTrainingProcessLauncher:
    def start(self, settings: SegmentationSettings, run_request: TrainingRunRequest) -> SpawnedTrainingProcessHandle:
        spawn_context = multiprocessing.get_context("spawn")
        event_queue: TrainingEventQueue = spawn_context.Queue()
        stop_event = spawn_context.Event()
        process = spawn_context.Process(
            target=run_training_process,
            args=(settings, run_request, event_queue, stop_event),
            name="training-run",
            daemon=False,
        )
        process.start()
        return SpawnedTrainingProcessHandle(process, event_queue, stop_event)
