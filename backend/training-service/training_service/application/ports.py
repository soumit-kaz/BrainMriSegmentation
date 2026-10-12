from typing import Any, Protocol

from segmentation_core.settings import SegmentationSettings

from training_service.domain.training_run import TrainingRunRequest


class TrainingProgressReporter(Protocol):
    def report(self, event: dict[str, Any]) -> None: ...

    def is_stop_requested(self) -> bool: ...


class TrainingHistoryStore(Protocol):
    epoch_records: list[dict[str, Any]]
    batch_points: list[dict[str, Any]]

    def load_until_epoch(self, last_epoch: int) -> None: ...

    def clear(self) -> None: ...

    def add_batch_point(self, batch_point: dict[str, Any]) -> None: ...

    def add_epoch_record(self, epoch_record: dict[str, Any]) -> None: ...


class TrainingProcessHandle(Protocol):
    def get_event(self, timeout_seconds: float) -> dict[str, Any] | None: ...

    def is_alive(self) -> bool: ...

    def get_exit_code(self) -> int | None: ...

    def request_stop(self) -> None: ...

    def stop(self, grace_seconds: float) -> None: ...


class TrainingProcessLauncher(Protocol):
    def start(self, settings: SegmentationSettings, run_request: TrainingRunRequest) -> TrainingProcessHandle: ...
