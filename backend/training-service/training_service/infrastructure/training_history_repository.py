from typing import Any

from segmentation_core.settings import SegmentationSettings, read_json_file, write_json_file


class JsonTrainingHistoryRepository:
    def __init__(self, settings: SegmentationSettings):
        self.file_path = settings.training_history_path
        self.epoch_records: list[dict[str, Any]] = []
        self.batch_points: list[dict[str, Any]] = []
        self.pending_batch_points: list[dict[str, Any]] = []

    def load_until_epoch(self, last_epoch: int) -> None:
        if not self.file_path.exists():
            return
        saved_history = read_json_file(self.file_path)
        self.epoch_records = [record for record in saved_history.get("epochs", []) if record["epoch"] <= last_epoch]
        self.batch_points = [point for point in saved_history.get("batches", []) if point["epoch"] <= last_epoch]

    def clear(self) -> None:
        self.epoch_records, self.batch_points, self.pending_batch_points = [], [], []
        self.save()

    def add_batch_point(self, batch_point: dict[str, Any]) -> None:
        self.pending_batch_points.append(batch_point)

    def add_epoch_record(self, epoch_record: dict[str, Any]) -> None:
        self.epoch_records.append(epoch_record)
        self.batch_points.extend(self.pending_batch_points)
        self.pending_batch_points = []
        self.save()

    def save(self) -> None:
        write_json_file(self.file_path, {"epochs": self.epoch_records, "batches": self.batch_points})
