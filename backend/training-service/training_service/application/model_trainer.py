import copy
import logging
import math
import random
import time
from typing import Any

import torch
from segmentation_core.data import (
    BratsCaseRepository,
    BratsSegmentationDataset,
    CaseCacheDataset,
    PreprocessedCaseProvider,
    get_augmented_batch,
    get_case_split,
)
from segmentation_core.evaluation import (
    SlidingWindowPredictor,
    get_dice_score,
    get_iou_score,
    get_metric_summary,
)
from segmentation_core.models import (
    ModelArchitecture,
    SegmentationModel,
    create_model,
    get_logits,
    get_parameter_count,
    load_model_checkpoint,
    save_model_checkpoint,
)
from segmentation_core.optimization import (
    DeepSupervisionLoss,
    ModelExponentialMovingAverage,
    TumorAwareSegmentationLoss,
)
from segmentation_core.runtime import (
    get_autocast_context,
    get_autocast_dtype,
    get_device,
    get_device_name,
    get_precision_name,
)
from segmentation_core.settings import (
    MODEL_NAME,
    SegmentationSettings,
    read_json_file,
    write_json_file,
)
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from training_service.application.ports import TrainingHistoryStore, TrainingProgressReporter
from training_service.domain.training_run import (
    EarlyStoppingMonitor,
    TrainingEventType,
    TrainingPhase,
    TrainingStoppedError,
)
from training_service.helpers import LearningRateSchedule, get_optimizer
from training_service.utils import get_utc_timestamp

logger = logging.getLogger("segmentation.training")

BATCH_LOG_INTERVAL = 50
GPU_MEMORY_HEADROOM_RATIO = 0.92
DATA_LOADER_PREFETCH_FACTOR = 4
PROBE_POSITIVE_VOXEL_RATIO = 0.1


class ModelTrainer:
    def __init__(self, settings: SegmentationSettings, progress_reporter: TrainingProgressReporter,
                 history_store: TrainingHistoryStore):
        self.settings = settings
        self.progress_reporter = progress_reporter
        self.history_store = history_store
        self.device = get_device()
        self.autocast_dtype = get_autocast_dtype(self.device, settings.is_mixed_precision_enabled)
        self.case_repository = BratsCaseRepository(settings.dataset_directory)
        self.case_provider = PreprocessedCaseProvider(self.case_repository, settings)

        self.model: SegmentationModel = create_model(ModelArchitecture.from_settings(settings)).to(self.device)
        self.loss_function = DeepSupervisionLoss(TumorAwareSegmentationLoss(settings))
        self.optimizer = get_optimizer(self.model, settings, self.device)
        self.gradient_scaler = GradScaler("cuda", enabled=self.autocast_dtype == torch.float16)
        self.model_average = ModelExponentialMovingAverage(self.model, settings.ema_decay) if settings.ema_decay > 0 else None
        self.early_stopping_monitor = EarlyStoppingMonitor(settings.early_stopping_patience)

        self.micro_batch_size = settings.batch_size
        self.accumulation_step_count = settings.gradient_accumulation_step_count
        self.start_epoch = 1

    @property
    def evaluation_model(self) -> SegmentationModel:
        return self.model_average.averaged_model if self.model_average else self.model

    def get_gpu_memory_megabytes(self) -> float | None:
        if self.device.type != "cuda":
            return None
        return round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1)

    def ensure_not_stopped(self) -> None:
        if self.progress_reporter.is_stop_requested():
            raise TrainingStoppedError()

    def report_progress(self, phase: TrainingPhase, epoch: int, step: int, step_count: int, **metrics: float) -> None:
        self.progress_reporter.report({
            "type": TrainingEventType.PROGRESS, "phase": phase, "epoch": epoch, "step": step, "step_count": step_count,
            "gpu_memory_megabytes": self.get_gpu_memory_megabytes(), **metrics,
        })

    def get_available_gpu_memory(self) -> int:
        free_memory, _ = torch.cuda.mem_get_info(self.device)
        return free_memory + torch.cuda.memory_reserved(self.device)

    def is_micro_batch_fitting(self, micro_batch_size: int, available_memory: int) -> bool:
        torch.cuda.reset_peak_memory_stats(self.device)
        self.model.train()
        probe_shape = (micro_batch_size, self.model.architecture.input_channel_count, *self.settings.patch_shape)
        probe_images = torch.randn(probe_shape, device=self.device)
        probe_masks = (torch.rand((micro_batch_size, 1, *self.settings.patch_shape), device=self.device) < PROBE_POSITIVE_VOXEL_RATIO).float()
        with get_autocast_context(self.device, self.autocast_dtype):
            probe_outputs = self.model(probe_images)
        self.loss_function(probe_outputs, probe_masks).backward()
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        logger.info("Memory probe: micro batch %d uses %.0f MB of %.0f MB available.", micro_batch_size, peak_memory / 2 ** 20, available_memory / 2 ** 20)
        return peak_memory <= available_memory * GPU_MEMORY_HEADROOM_RATIO

    def get_fitted_micro_batch_size(self) -> int:
        if self.device.type != "cuda" or not self.settings.is_auto_batch_fitting_enabled:
            return self.settings.batch_size
        saved_model_state = copy.deepcopy(self.model.state_dict())
        available_memory = self.get_available_gpu_memory()
        micro_batch_size = self.settings.batch_size
        while micro_batch_size >= 1:
            try:
                is_fitting = self.is_micro_batch_fitting(micro_batch_size, available_memory)
            except torch.cuda.OutOfMemoryError:
                logger.warning("Memory probe: micro batch %d does not fit.", micro_batch_size)
                is_fitting = False
            finally:
                self.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                self.model.load_state_dict(saved_model_state)
            if is_fitting:
                return micro_batch_size
            micro_batch_size //= 2
        raise RuntimeError("The model does not fit in GPU memory even with micro batch 1. "
                           "Enable gradient checkpointing or reduce patch_shape or encoder_channel_counts.")

    def get_new_case_split(self) -> dict[str, list[str]]:
        case_names = self.case_repository.get_case_names()
        if self.settings.maximum_case_count > 0:
            selected_count = min(self.settings.maximum_case_count, len(case_names))
            case_names = random.Random(self.settings.random_seed).sample(case_names, selected_count)
        case_split = get_case_split(case_names, self.settings)
        write_json_file(self.settings.data_split_path, case_split)
        return case_split

    def get_saved_case_split(self) -> dict[str, list[str]]:
        if not self.settings.data_split_path.exists():
            raise FileNotFoundError(f"No data split found at {self.settings.data_split_path}. Train first.")
        return read_json_file(self.settings.data_split_path)

    def get_data_loader_options(self) -> dict[str, Any]:
        worker_count = self.settings.data_loader_worker_count
        options: dict[str, Any] = {"num_workers": worker_count, "pin_memory": self.device.type == "cuda"}
        if worker_count > 0:
            options.update(persistent_workers=True, prefetch_factor=DATA_LOADER_PREFETCH_FACTOR)
        return options

    def get_training_loader(self, case_names: list[str]) -> DataLoader[Any]:
        if self.settings.iterations_per_epoch > 0:
            sample_count = self.settings.iterations_per_epoch * self.micro_batch_size * self.accumulation_step_count
        else:
            sample_count = max(len(case_names), self.micro_batch_size)
        dataset = BratsSegmentationDataset(self.case_provider, case_names, self.settings, is_training=True, sample_count=sample_count)
        return DataLoader(dataset, batch_size=self.micro_batch_size, shuffle=True, drop_last=True, **self.get_data_loader_options())

    def get_evaluation_loader(self, case_names: list[str]) -> DataLoader[Any]:
        dataset = BratsSegmentationDataset(self.case_provider, case_names, self.settings, is_training=False)
        return DataLoader(dataset, batch_size=1, shuffle=False, **self.get_data_loader_options())

    def build_case_cache(self, case_names: list[str]) -> None:
        if not self.settings.is_case_cache_enabled:
            return
        uncached_case_names = [case_name for case_name in case_names if not self.case_provider.is_cached(case_name)]
        if not uncached_case_names:
            return
        logger.info("Preprocessing %d cases into the cache at %s.", len(uncached_case_names), self.case_provider.cache_directory)
        cache_loader = DataLoader(CaseCacheDataset(self.case_provider, uncached_case_names), batch_size=1,
                                  num_workers=self.settings.data_loader_worker_count)
        for case_index, _ in enumerate(cache_loader, start=1):
            self.ensure_not_stopped()
            self.report_progress(TrainingPhase.CACHING, 0, case_index, len(uncached_case_names))

    def get_prepared_training_loader(self, training_case_names: list[str]) -> tuple[DataLoader[Any], LambdaLR]:
        self.micro_batch_size = self.get_fitted_micro_batch_size()
        self.accumulation_step_count = max(self.settings.gradient_accumulation_step_count,
                                           math.ceil(self.settings.batch_size / self.micro_batch_size))
        training_loader = self.get_training_loader(training_case_names)
        steps_per_epoch = math.ceil(len(training_loader) / self.accumulation_step_count)
        learning_rate_scheduler = LambdaLR(self.optimizer, LearningRateSchedule(self.settings, steps_per_epoch))
        return training_loader, learning_rate_scheduler

    def save_training_state(self, epoch: int, learning_rate_scheduler: LambdaLR) -> None:
        temporary_path = self.settings.last_checkpoint_path.with_suffix(".tmp")
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": learning_rate_scheduler.state_dict(),
            "scaler_state": self.gradient_scaler.state_dict(),
            "average_state": self.model_average.get_state() if self.model_average else None,
            "early_stopping_state": self.early_stopping_monitor.get_state(),
        }, temporary_path)
        temporary_path.replace(self.settings.last_checkpoint_path)

    def load_training_state(self, learning_rate_scheduler: LambdaLR) -> None:
        if not self.settings.last_checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint to resume from at {self.settings.last_checkpoint_path}.")
        training_state = torch.load(self.settings.last_checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(training_state["model_state"])
        self.optimizer.load_state_dict(training_state["optimizer_state"])
        learning_rate_scheduler.load_state_dict(training_state["scheduler_state"])
        self.gradient_scaler.load_state_dict(training_state["scaler_state"])
        if self.model_average and training_state["average_state"]:
            self.model_average.set_state(training_state["average_state"])
        self.early_stopping_monitor.set_state(training_state["early_stopping_state"])
        self.history_store.load_until_epoch(training_state["epoch"])
        self.start_epoch = training_state["epoch"] + 1
        logger.info("Resumed from epoch %d.", training_state["epoch"])

    def get_training_progress(self, epoch: int) -> float:
        return (epoch - 1) / max(1, self.settings.epoch_count - 1)

    def is_augmentation_active(self, epoch: int) -> bool:
        return self.settings.is_augmentation_enabled and epoch <= self.settings.epoch_count - self.settings.augmentation_close_epoch_count

    def run_optimizer_step(self, learning_rate_scheduler: LambdaLR) -> None:
        if self.settings.maximum_gradient_norm > 0:
            self.gradient_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings.maximum_gradient_norm)
        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        learning_rate_scheduler.step()
        if self.model_average:
            self.model_average.update(self.model)

    def get_training_epoch_metrics(self, training_loader: DataLoader[Any], learning_rate_scheduler: LambdaLR,
                                   epoch: int) -> dict[str, float]:
        self.model.train()
        is_augmentation_active = self.is_augmentation_active(epoch)
        loss_total, dice_total, batch_count = 0.0, 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, masks, _) in enumerate(training_loader, start=1):
            self.ensure_not_stopped()
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            if is_augmentation_active:
                images, masks = get_augmented_batch(images, masks, self.settings.modality_dropout_probability)

            with get_autocast_context(self.device, self.autocast_dtype):
                outputs = self.model(images)
            loss = self.loss_function(outputs, masks)
            self.gradient_scaler.scale(loss / self.accumulation_step_count).backward()
            if batch_index % self.accumulation_step_count == 0 or batch_index == len(training_loader):
                self.run_optimizer_step(learning_rate_scheduler)

            predicted_masks = torch.sigmoid(get_logits(outputs).detach().float()) >= self.settings.prediction_threshold
            batch_loss = loss.item()
            batch_dice = get_dice_score(predicted_masks, masks >= 0.5)
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            loss_total += batch_loss
            dice_total += batch_dice
            batch_count += 1
            self.history_store.add_batch_point({
                "epoch": epoch, "step": batch_index, "step_count": len(training_loader),
                "loss": batch_loss, "dice": batch_dice, "learning_rate": learning_rate,
            })
            self.report_progress(TrainingPhase.TRAINING, epoch, batch_index, len(training_loader),
                                 loss=batch_loss, dice=batch_dice, learning_rate=learning_rate)
            if batch_index % BATCH_LOG_INTERVAL == 0:
                logger.info("Epoch %d batch %d/%d loss %.4f", epoch, batch_index, len(training_loader), loss_total / batch_count)
        return {"loss": loss_total / batch_count, "dice": dice_total / batch_count}

    def get_case_evaluation_records(self, model: SegmentationModel, evaluation_loader: DataLoader[Any],
                                    phase: TrainingPhase, epoch: int) -> list[dict[str, Any]]:
        predictor = SlidingWindowPredictor(model, self.settings.patch_shape, self.device, self.autocast_dtype,
                                           self.settings.sliding_window_batch_size)
        case_records: list[dict[str, Any]] = []
        for case_index, (image, mask, case_names) in enumerate(evaluation_loader, start=1):
            self.ensure_not_stopped()
            logit_volume = predictor.get_logit_volume(image[0])
            target_mask = mask[0, 0].to(self.device)
            predicted_mask = torch.sigmoid(logit_volume) >= self.settings.prediction_threshold
            case_record: dict[str, Any] = {
                "case_name": case_names[0],
                "loss": self.loss_function(logit_volume[None, None], target_mask[None, None]).item(),
                "dice": get_dice_score(predicted_mask, target_mask >= 0.5),
                "iou": get_iou_score(predicted_mask, target_mask >= 0.5),
            }
            case_records.append(case_record)
            self.report_progress(phase, epoch, case_index, len(evaluation_loader), loss=case_record["loss"], dice=case_record["dice"])
        return case_records

    @staticmethod
    def get_average_metrics(case_records: list[dict[str, Any]]) -> dict[str, float]:
        return {metric_name: sum(record[metric_name] for record in case_records) / len(case_records)
                for metric_name in ("loss", "dice", "iou")}

    def get_run_information(self, case_split: dict[str, list[str]], steps_per_epoch: int) -> dict[str, Any]:
        return {
            "model_name": MODEL_NAME,
            "device": self.device.type,
            "device_name": get_device_name(self.device),
            "precision": get_precision_name(self.autocast_dtype),
            "parameter_count": get_parameter_count(self.model),
            "training_case_count": len(case_split["training"]),
            "validation_case_count": len(case_split["validation"]),
            "test_case_count": len(case_split["test"]),
            "start_epoch": self.start_epoch,
            "epoch_count": self.settings.epoch_count,
            "batch_size": self.micro_batch_size * self.accumulation_step_count,
            "micro_batch_size": self.micro_batch_size,
            "accumulation_step_count": self.accumulation_step_count,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": self.settings.learning_rate,
            "is_ema_enabled": self.model_average is not None,
            "is_gradient_checkpointing_enabled": self.settings.is_gradient_checkpointing_enabled,
            "is_augmentation_enabled": self.settings.is_augmentation_enabled,
            "modality_dropout_probability": self.settings.modality_dropout_probability,
            "early_stopping_patience": self.settings.early_stopping_patience,
        }

    def log_run_information(self, run_information: dict[str, Any]) -> None:
        logger.info("Model %s (%s parameters) | device %s | precision %s", run_information["model_name"],
                    f"{run_information['parameter_count']:,}", run_information["device_name"], run_information["precision"])
        logger.info("Batch %d = micro batch %d x %d accumulation | %d optimizer steps per epoch",
                    run_information["batch_size"], self.micro_batch_size, self.accumulation_step_count, run_information["steps_per_epoch"])
        logger.info("Cases: training %d | validation %d | test %d", run_information["training_case_count"],
                    run_information["validation_case_count"], run_information["test_case_count"])

    def get_epoch_record(self, epoch: int, learning_rate: float, training_metrics: dict[str, float],
                         validation_metrics: dict[str, float], is_best_epoch: bool, epoch_started_at: float) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "training_loss": training_metrics["loss"],
            "training_dice": training_metrics["dice"],
            "validation_loss": validation_metrics["loss"],
            "validation_dice": validation_metrics["dice"],
            "validation_iou": validation_metrics["iou"],
            "is_best_epoch": is_best_epoch,
            "is_augmentation_active": self.is_augmentation_active(epoch),
            "gpu_memory_megabytes": self.get_gpu_memory_megabytes(),
            "duration_seconds": round(time.perf_counter() - epoch_started_at, 1),
        }

    def run_training(self, is_resume_requested: bool) -> None:
        case_split = self.get_saved_case_split() if is_resume_requested else self.get_new_case_split()
        self.build_case_cache(case_split["training"] + case_split["validation"] + case_split["test"])
        training_loader, learning_rate_scheduler = self.get_prepared_training_loader(case_split["training"])
        if is_resume_requested:
            self.load_training_state(learning_rate_scheduler)
        else:
            self.history_store.clear()

        run_information = self.get_run_information(case_split, math.ceil(len(training_loader) / self.accumulation_step_count))
        self.progress_reporter.report({"type": TrainingEventType.RUN_STARTED, "run": run_information})
        self.log_run_information(run_information)
        validation_loader = self.get_evaluation_loader(case_split["validation"])
        training_started_at = time.perf_counter()

        for epoch in range(self.start_epoch, self.settings.epoch_count + 1):
            epoch_started_at = time.perf_counter()
            learning_rate = float(self.optimizer.param_groups[0]["lr"])
            self.loss_function.set_training_progress(self.get_training_progress(epoch))
            training_metrics = self.get_training_epoch_metrics(training_loader, learning_rate_scheduler, epoch)
            validation_records = self.get_case_evaluation_records(self.evaluation_model, validation_loader, TrainingPhase.VALIDATION, epoch)
            validation_metrics = self.get_average_metrics(validation_records)

            is_best_epoch = self.early_stopping_monitor.is_improved(validation_metrics["dice"])
            if is_best_epoch:
                save_model_checkpoint(self.settings.best_model_path, self.evaluation_model, self.settings, epoch,
                                      {f"validation_{name}": value for name, value in validation_metrics.items()})

            epoch_record = self.get_epoch_record(epoch, learning_rate, training_metrics, validation_metrics, is_best_epoch, epoch_started_at)
            self.history_store.add_epoch_record(epoch_record)
            self.save_training_state(epoch, learning_rate_scheduler)
            self.progress_reporter.report({"type": TrainingEventType.EPOCH, "record": epoch_record})
            logger.info(
                "Epoch %d/%d | lr %.2e | train loss %.4f dice %.4f | val loss %.4f dice %.4f iou %.4f%s",
                epoch, self.settings.epoch_count, learning_rate, training_metrics["loss"], training_metrics["dice"],
                validation_metrics["loss"], validation_metrics["dice"], validation_metrics["iou"], " | best" if is_best_epoch else "",
            )
            if self.early_stopping_monitor.is_stop_required:
                logger.info("Early stopping: no improvement for %d epochs.", self.settings.early_stopping_patience)
                break

        logger.info("Training finished in %.1f minutes.", (time.perf_counter() - training_started_at) / 60)
        self.run_test_evaluation()

    def run_test_evaluation(self) -> None:
        if not self.settings.best_model_path.exists():
            raise FileNotFoundError(f"No trained model at {self.settings.best_model_path}.")
        best_model, best_checkpoint = load_model_checkpoint(self.settings.best_model_path, self.device)
        logger.info("Evaluating best model (epoch %d) on the test split.", best_checkpoint["epoch"])

        test_case_names = self.get_saved_case_split()["test"]
        self.build_case_cache(test_case_names)
        test_loader = self.get_evaluation_loader(test_case_names)
        case_records = self.get_case_evaluation_records(best_model, test_loader, TrainingPhase.TEST, best_checkpoint["epoch"])
        test_report = {
            "evaluated_at": get_utc_timestamp(),
            "model_epoch": best_checkpoint["epoch"],
            "model_name": best_checkpoint["model_name"],
            "case_count": len(case_records),
            "dice": get_metric_summary([record["dice"] for record in case_records]),
            "iou": get_metric_summary([record["iou"] for record in case_records]),
            "cases": case_records,
        }
        write_json_file(self.settings.test_report_path, test_report)
        self.progress_reporter.report({"type": TrainingEventType.TEST_REPORT, "report": test_report})
        for metric_name in ("dice", "iou"):
            summary = test_report[metric_name]
            logger.info("Test %-4s | mean %.4f | median %.4f | std %.4f | 25%% %.4f | 75%% %.4f", metric_name,
                        summary["mean"], summary["median"], summary["standard_deviation"],
                        summary["percentile_25"], summary["percentile_75"])
