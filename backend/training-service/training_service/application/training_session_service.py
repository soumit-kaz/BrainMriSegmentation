import asyncio
import copy
import threading
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any

from segmentation_core.data import BratsCaseRepository
from segmentation_core.settings import SegmentationSettings, read_json_file

from training_service.application.ports import TrainingProcessHandle, TrainingProcessLauncher
from training_service.domain.training_run import (
    TERMINAL_TRAINING_STATES,
    TrainingCommand,
    TrainingConflictError,
    TrainingEventType,
    TrainingPhase,
    TrainingRunRequest,
    TrainingState,
)
from training_service.utils import get_server_sent_event, get_server_sent_heartbeat, get_utc_timestamp

MAXIMUM_LOG_COUNT = 400
MAXIMUM_BATCH_POINT_COUNT = 100_000
SUBSCRIBER_QUEUE_SIZE = 2000
HEARTBEAT_SECONDS = 15.0
EVENT_POLL_SECONDS = 1.0
SHUTDOWN_GRACE_SECONDS = 30.0
BATCH_POINT_FIELDS = ("epoch", "step", "step_count", "loss", "dice", "learning_rate")
LOG_FIELDS = ("timestamp", "level", "message")


class TrainingSessionService:
    def __init__(self, settings: SegmentationSettings, process_launcher: TrainingProcessLauncher):
        self.settings = settings
        self.process_launcher = process_launcher
        self.lock = threading.Lock()
        self.process_handle: TrainingProcessHandle | None = None
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.logs: deque[dict[str, Any]] = deque(maxlen=MAXIMUM_LOG_COUNT)
        saved_history = self.get_saved_history()
        self.batch_points: deque[dict[str, Any]] = deque(saved_history.get("batches", []), maxlen=MAXIMUM_BATCH_POINT_COUNT)
        self.snapshot: dict[str, Any] = {
            "state": TrainingState.IDLE,
            "command": None,
            "message": None,
            "started_at": None,
            "finished_at": None,
            "run": None,
            "progress": None,
            "epochs": saved_history.get("epochs", []),
            "test_report": self.get_saved_test_report(),
        }

    def attach_event_loop(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def get_saved_history(self) -> dict[str, Any]:
        if not self.settings.training_history_path.exists():
            return {}
        return read_json_file(self.settings.training_history_path)

    def get_saved_test_report(self) -> dict[str, Any] | None:
        if not self.settings.test_report_path.exists():
            return None
        return read_json_file(self.settings.test_report_path)

    def get_dataset_case_count(self) -> int:
        return len(BratsCaseRepository(self.settings.dataset_directory).get_case_names())

    def get_snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy({**self.snapshot, "logs": list(self.logs), "batch_points": list(self.batch_points)})

    def is_running(self) -> bool:
        return self.snapshot["state"] not in TERMINAL_TRAINING_STATES

    def validate_run_request(self, run_request: TrainingRunRequest) -> None:
        if run_request.command == TrainingCommand.EVALUATE and not self.settings.best_model_path.exists():
            raise TrainingConflictError("No trained model to evaluate yet.")
        if run_request.is_resume_requested and not self.settings.last_checkpoint_path.exists():
            raise TrainingConflictError("No checkpoint to resume from.")

    def start_run(self, run_request: TrainingRunRequest) -> dict[str, Any]:
        self.validate_run_request(run_request)
        run_settings = SegmentationSettings.from_environment().get_updated(**run_request.setting_overrides)
        with self.lock:
            if self.is_running():
                raise TrainingConflictError("A training run is already in progress.")
            saved_history = self.get_saved_history() if run_request.is_history_kept else {}
            self.logs.clear()
            self.batch_points = deque(saved_history.get("batches", []), maxlen=MAXIMUM_BATCH_POINT_COUNT)
            self.snapshot.update({
                "state": TrainingState.STARTING,
                "command": run_request.command,
                "message": None,
                "started_at": get_utc_timestamp(),
                "finished_at": None,
                "run": None,
                "progress": None,
                "epochs": saved_history.get("epochs", []),
                "test_report": None if run_request.command == TrainingCommand.TRAIN else self.snapshot["test_report"],
            })
            process_handle = self.process_launcher.start(run_settings, run_request)
            self.process_handle = process_handle
        threading.Thread(target=self.pump_events, args=(process_handle,), name="training-event-pump", daemon=True).start()
        snapshot = self.get_snapshot()
        self.broadcast({"type": TrainingEventType.SNAPSHOT, "snapshot": snapshot})
        return snapshot

    def request_stop(self) -> dict[str, Any]:
        with self.lock:
            if not self.is_running() or self.process_handle is None:
                raise TrainingConflictError("No training run is in progress.")
            self.process_handle.request_stop()
            self.snapshot["state"] = TrainingState.STOPPING
        self.broadcast({"type": TrainingEventType.STATE, "state": TrainingState.STOPPING})
        return self.get_snapshot()

    def shutdown(self) -> None:
        if self.process_handle is not None and self.process_handle.is_alive():
            self.process_handle.stop(SHUTDOWN_GRACE_SECONDS)

    def pump_events(self, process_handle: TrainingProcessHandle) -> None:
        while True:
            event = process_handle.get_event(EVENT_POLL_SECONDS)
            if event is not None:
                self.add_event(event)
            elif not process_handle.is_alive():
                break
        with self.lock:
            is_unfinished = self.is_running()
        if is_unfinished:
            self.add_event({
                "type": TrainingEventType.RUN_FINISHED,
                "state": TrainingState.FAILED,
                "timestamp": get_utc_timestamp(),
                "message": f"Training process exited unexpectedly (exit code {process_handle.get_exit_code()}).",
            })

    def add_event(self, event: dict[str, Any]) -> None:
        event_type = event["type"]
        with self.lock:
            if event_type == TrainingEventType.LOG:
                self.logs.append({field_name: event[field_name] for field_name in LOG_FIELDS})
            elif event_type == TrainingEventType.RUN_STARTED:
                self.snapshot.update({"state": TrainingState.RUNNING, "run": event["run"]})
            elif event_type == TrainingEventType.PROGRESS:
                event = self.get_progress_event_with_batch_point(event)
            elif event_type == TrainingEventType.EPOCH:
                self.snapshot["epochs"].append(event["record"])
            elif event_type == TrainingEventType.TEST_REPORT:
                self.snapshot["test_report"] = event["report"]
            elif event_type == TrainingEventType.RUN_FINISHED:
                self.snapshot.update({"state": event["state"], "message": event["message"], "finished_at": event["timestamp"]})
        self.broadcast(event)

    def get_progress_event_with_batch_point(self, event: dict[str, Any]) -> dict[str, Any]:
        self.snapshot["progress"] = event
        if self.snapshot["state"] == TrainingState.STARTING:
            self.snapshot["state"] = TrainingState.RUNNING
        if event["phase"] != TrainingPhase.TRAINING:
            return event
        batch_point = {field_name: event[field_name] for field_name in BATCH_POINT_FIELDS}
        self.batch_points.append(batch_point)
        return {**event, "batch_point": batch_point}

    def broadcast(self, event: dict[str, Any]) -> None:
        if self.event_loop is None:
            return
        for subscriber_queue in list(self.subscribers):
            self.event_loop.call_soon_threadsafe(self.add_to_subscriber_queue, subscriber_queue, event)

    @staticmethod
    def add_to_subscriber_queue(subscriber_queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        if not subscriber_queue.full():
            subscriber_queue.put_nowait(event)

    async def get_event_stream(self) -> AsyncGenerator[str, None]:
        subscriber_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        self.subscribers.add(subscriber_queue)
        try:
            yield get_server_sent_event({"type": TrainingEventType.SNAPSHOT, "snapshot": self.get_snapshot()})
            while True:
                try:
                    event = await asyncio.wait_for(subscriber_queue.get(), timeout=HEARTBEAT_SECONDS)
                    yield get_server_sent_event(event)
                except asyncio.TimeoutError:
                    yield get_server_sent_heartbeat()
        finally:
            self.subscribers.discard(subscriber_queue)
