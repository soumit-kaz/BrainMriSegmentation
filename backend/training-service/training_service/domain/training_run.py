from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TrainingState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


TERMINAL_TRAINING_STATES: frozenset[TrainingState] = frozenset(
    {TrainingState.IDLE, TrainingState.COMPLETED, TrainingState.STOPPED, TrainingState.FAILED}
)


class TrainingCommand(StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"


class TrainingEventType(StrEnum):
    SNAPSHOT = "snapshot"
    STATE = "state"
    LOG = "log"
    RUN_STARTED = "run_started"
    PROGRESS = "progress"
    EPOCH = "epoch"
    TEST_REPORT = "test_report"
    RUN_FINISHED = "run_finished"


class TrainingPhase(StrEnum):
    CACHING = "caching"
    TRAINING = "training"
    VALIDATION = "validation"
    TEST = "test"


class TrainingConflictError(Exception):
    pass


class TrainingStoppedError(Exception):
    pass


@dataclass(frozen=True)
class TrainingRunRequest:
    command: TrainingCommand
    is_resume_requested: bool
    setting_overrides: dict[str, Any]

    @property
    def is_history_kept(self) -> bool:
        return self.is_resume_requested or self.command == TrainingCommand.EVALUATE


class EarlyStoppingMonitor:
    def __init__(self, patience: int):
        self.patience = patience
        self.best_score = float("-inf")
        self.epochs_without_improvement = 0

    def is_improved(self, score: float) -> bool:
        if score > self.best_score:
            self.best_score = score
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def is_stop_required(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def get_state(self) -> dict[str, Any]:
        return {"best_score": self.best_score, "epochs_without_improvement": self.epochs_without_improvement}

    def set_state(self, state: dict[str, Any]) -> None:
        self.best_score = state["best_score"]
        self.epochs_without_improvement = state["epochs_without_improvement"]
