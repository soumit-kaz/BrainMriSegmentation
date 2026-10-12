from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.checkpoint import checkpoint

from segmentation_core.settings import MODALITY_SUFFIXES, MODEL_NAME, SegmentationSettings

ModelOutput = torch.Tensor | list[torch.Tensor]
SPATIAL_AXIS_COUNT = 3


@dataclass(frozen=True)
class ModelArchitecture:
    input_channel_count: int = len(MODALITY_SUFFIXES)
    output_channel_count: int = 1
    modality_channel_count: int = 8
    attention_reduction_ratio: int = 8
    spatial_attention_kernel_size: int = 7
    encoder_channel_counts: tuple[int, ...] = (24, 48, 96, 192, 320)
    stage_depths: tuple[int, ...] = (1, 1, 2, 2, 1)
    attention_block_count: int = 3
    area_count: int = 4
    attention_head_dimension: int = 32
    drop_path_rate: float = 0.1
    is_deep_supervision_enabled: bool = True
    is_gradient_checkpointing_enabled: bool = True

    @classmethod
    def from_settings(cls, settings: SegmentationSettings) -> "ModelArchitecture":
        settings_values: dict[str, Any] = {architecture_field.name: getattr(settings, architecture_field.name)
                                           for architecture_field in fields(cls) if hasattr(settings, architecture_field.name)}
        return cls(**settings_values)

    @classmethod
    def from_dictionary(cls, values: dict[str, Any]) -> "ModelArchitecture":
        known_names = {architecture_field.name for architecture_field in fields(cls)}
        normalized_values: dict[str, Any] = {name: tuple(value) if isinstance(value, list) else value
                                             for name, value in values.items() if name in known_names}
        return cls(**normalized_values)


class ConvNormActivation(nn.Sequential):
    def __init__(self, input_channel_count: int, output_channel_count: int, kernel_size: int = 3, stride: int = 1, group_count: int = 1):
        super().__init__(
            nn.Conv3d(input_channel_count, output_channel_count, kernel_size, stride, padding=kernel_size // 2,
                      groups=group_count, bias=False),
            nn.InstanceNorm3d(output_channel_count, affine=True),
            nn.SiLU(inplace=True),
        )


class ConvNorm(nn.Sequential):
    def __init__(self, input_channel_count: int, output_channel_count: int, kernel_size: int = 1):
        super().__init__(
            nn.Conv3d(input_channel_count, output_channel_count, kernel_size, padding=kernel_size // 2, bias=False),
            nn.InstanceNorm3d(output_channel_count, affine=True),
        )


class ChannelAttention(nn.Module):
    def __init__(self, channel_count: int, reduction_ratio: int):
        super().__init__()
        hidden_channel_count = max(channel_count // reduction_ratio, 1)
        self.shared_perceptron = nn.Sequential(
            nn.Conv3d(channel_count, hidden_channel_count, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channel_count, channel_count, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        average_descriptor = functional.adaptive_avg_pool3d(features, 1)
        maximum_descriptor = functional.adaptive_max_pool3d(features, 1)
        return features * torch.sigmoid(self.shared_perceptron(average_descriptor) + self.shared_perceptron(maximum_descriptor))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.convolution = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled_maps = torch.cat([features.mean(dim=1, keepdim=True), features.amax(dim=1, keepdim=True)], dim=1)
        return features * torch.sigmoid(self.convolution(pooled_maps))


class SkipAttentionGate(nn.Sequential):
    def __init__(self, channel_count: int, reduction_ratio: int, kernel_size: int):
        super().__init__(ChannelAttention(channel_count, reduction_ratio), SpatialAttention(kernel_size))


class ModalityAwareStem(nn.Module):
    def __init__(self, architecture: ModelArchitecture, output_channel_count: int):
        super().__init__()
        self.modality_count = architecture.input_channel_count
        self.modality_channel_count = architecture.modality_channel_count
        modality_feature_count = self.modality_count * self.modality_channel_count
        self.modality_encoders = nn.Sequential(
            ConvNormActivation(self.modality_count, modality_feature_count, group_count=self.modality_count),
            ConvNormActivation(modality_feature_count, modality_feature_count, group_count=self.modality_count),
        )
        hidden_channel_count = max(modality_feature_count // architecture.attention_reduction_ratio, self.modality_count)
        self.modality_exchange = nn.Sequential(
            nn.Conv3d(modality_feature_count, hidden_channel_count, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channel_count, modality_feature_count, kernel_size=1),
        )
        self.fusion = ConvNormActivation(modality_feature_count, output_channel_count, kernel_size=1)
        self.joint_projection = ConvNorm(self.modality_count, output_channel_count, kernel_size=3)

    def get_modality_weights(self, modality_features: torch.Tensor) -> torch.Tensor:
        batch_size = modality_features.shape[0]
        exchange_logits = self.modality_exchange(functional.adaptive_avg_pool3d(modality_features, 1))
        exchange_logits = exchange_logits.view(batch_size, self.modality_count, self.modality_channel_count)
        modality_weights = torch.softmax(exchange_logits, dim=1) * self.modality_count
        return modality_weights.view(batch_size, self.modality_count * self.modality_channel_count, 1, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        modality_features = self.modality_encoders(image)
        weighted_features = modality_features * self.get_modality_weights(modality_features)
        return functional.silu(self.fusion(weighted_features) + self.joint_projection(image))


class ResidualDownsample(nn.Module):
    def __init__(self, input_channel_count: int, output_channel_count: int):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv3d(input_channel_count, output_channel_count, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(output_channel_count, affine=True),
        )
        self.skip_path = nn.Sequential(nn.MaxPool3d(kernel_size=2), ConvNorm(input_channel_count, output_channel_count))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.silu(self.main_path(features) + self.skip_path(features))


class DropPath(nn.Module):
    def __init__(self, drop_probability: float):
        super().__init__()
        self.drop_probability = drop_probability

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_probability == 0.0:
            return features
        keep_probability = 1.0 - self.drop_probability
        keep_mask = features.new_empty((features.shape[0],) + (1,) * (features.ndim - 1)).bernoulli_(keep_probability)
        return features * keep_mask / keep_probability


class LayerScale(nn.Module):
    def __init__(self, channel_count: int, initial_value: float = 1e-2):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1, channel_count, 1, 1, 1), initial_value))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features * self.scale


class ResidualBottleneck(nn.Module):
    def __init__(self, channel_count: int, drop_path_rate: float):
        super().__init__()
        self.convolutions = nn.Sequential(ConvNormActivation(channel_count, channel_count), ConvNorm(channel_count, channel_count, kernel_size=3))
        self.layer_scale = LayerScale(channel_count, initial_value=1.0)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return functional.silu(features + self.drop_path(self.layer_scale(self.convolutions(features))))


class CrossStagePartialBlock(nn.Module):
    def __init__(self, input_channel_count: int, output_channel_count: int, drop_path_rates: list[float]):
        super().__init__()
        hidden_channel_count = max(output_channel_count // 2, 8)
        self.input_projection = ConvNormActivation(input_channel_count, 2 * hidden_channel_count, kernel_size=1)
        self.bottlenecks = nn.ModuleList(ResidualBottleneck(hidden_channel_count, rate) for rate in drop_path_rates)
        self.output_projection = ConvNormActivation((2 + len(drop_path_rates)) * hidden_channel_count, output_channel_count, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        first_half, second_half = self.input_projection(features).chunk(2, dim=1)
        collected_features = [first_half, second_half]
        for bottleneck in self.bottlenecks:
            second_half = bottleneck(second_half)
            collected_features.append(second_half)
        return self.output_projection(torch.cat(collected_features, dim=1))


class SpatialPyramidPoolingFast(nn.Module):
    POOLING_REPEAT_COUNT = 3

    def __init__(self, channel_count: int):
        super().__init__()
        hidden_channel_count = channel_count // 2
        self.input_projection = ConvNormActivation(channel_count, hidden_channel_count, kernel_size=1)
        self.pooling = nn.MaxPool3d(kernel_size=5, stride=1, padding=2)
        self.output_projection = ConvNormActivation(hidden_channel_count * (self.POOLING_REPEAT_COUNT + 1), channel_count, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled_features = [self.input_projection(features)]
        for _ in range(self.POOLING_REPEAT_COUNT):
            pooled_features.append(self.pooling(pooled_features[-1]))
        return self.output_projection(torch.cat(pooled_features, dim=1))


class AxisRotatingAreaAttention(nn.Module):
    def __init__(self, channel_count: int, head_dimension: int, area_count: int, slab_axis: int):
        super().__init__()
        self.head_count = channel_count // head_dimension
        self.area_count = area_count
        self.slab_axis = slab_axis
        self.query_key_value = nn.Conv3d(channel_count, channel_count * 3, kernel_size=1, bias=False)
        self.position_perceiver = nn.Conv3d(channel_count, channel_count, kernel_size=5, padding=2, groups=channel_count)
        self.output_projection = nn.Conv3d(channel_count, channel_count, kernel_size=1)

    def get_area_count(self, slab_axis_size: int) -> int:
        return self.area_count if self.area_count > 1 and slab_axis_size % self.area_count == 0 else 1

    def get_head_tokens(self, slab_first_map: torch.Tensor, area_count: int) -> torch.Tensor:
        batch_size, channel_count = slab_first_map.shape[:2]
        token_count = slab_first_map[0, 0].numel() // area_count
        tokens = slab_first_map.flatten(2).transpose(1, 2).reshape(batch_size * area_count, token_count, channel_count)
        return tokens.reshape(batch_size * area_count, token_count, self.head_count, -1).transpose(1, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        slab_dimension = 2 + self.slab_axis
        query_map, key_map, value_map = self.query_key_value(features).chunk(3, dim=1)
        slab_first_query = query_map.movedim(slab_dimension, 2)
        slab_first_shape = slab_first_query.shape
        area_count = self.get_area_count(slab_first_shape[2])
        attended_tokens = functional.scaled_dot_product_attention(
            self.get_head_tokens(slab_first_query, area_count),
            self.get_head_tokens(key_map.movedim(slab_dimension, 2), area_count),
            self.get_head_tokens(value_map.movedim(slab_dimension, 2), area_count),
        )
        batch_size, channel_count = slab_first_shape[:2]
        attended_map = (attended_tokens.transpose(1, 2).reshape(batch_size, -1, channel_count).transpose(1, 2)
                        .reshape(slab_first_shape).movedim(2, slab_dimension))
        return self.output_projection(attended_map + self.position_perceiver(value_map))


class AreaAttentionBlock(nn.Module):
    def __init__(self, channel_count: int, head_dimension: int, area_count: int, slab_axis: int, drop_path_rate: float):
        super().__init__()
        self.attention_norm = nn.GroupNorm(1, channel_count)
        self.attention = AxisRotatingAreaAttention(channel_count, head_dimension, area_count, slab_axis)
        self.attention_scale = LayerScale(channel_count)
        self.perceptron_norm = nn.GroupNorm(1, channel_count)
        self.perceptron = nn.Sequential(
            nn.Conv3d(channel_count, channel_count * 2, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(channel_count * 2, channel_count, kernel_size=1),
        )
        self.perceptron_scale = LayerScale(channel_count)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features + self.drop_path(self.attention_scale(self.attention(self.attention_norm(features))))
        return features + self.drop_path(self.perceptron_scale(self.perceptron(self.perceptron_norm(features))))


class DecoderStage(nn.Module):
    def __init__(self, input_channel_count: int, output_channel_count: int, architecture: ModelArchitecture):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(input_channel_count, output_channel_count, kernel_size=2, stride=2)
        self.skip_gate = SkipAttentionGate(output_channel_count, architecture.attention_reduction_ratio, architecture.spatial_attention_kernel_size)
        self.fusion = CrossStagePartialBlock(2 * output_channel_count, output_channel_count, [0.0])

    def forward(self, features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.upsample(features), self.skip_gate(skip_features)], dim=1))


class DropPathRateSchedule:
    def __init__(self, maximum_rate: float, total_count: int):
        self.rates = torch.linspace(0, maximum_rate, max(total_count, 1)).tolist()
        self.next_index = 0

    def get_next_rates(self, count: int) -> list[float]:
        selected_rates = self.rates[self.next_index:self.next_index + count]
        self.next_index += count
        return selected_rates


class ModalityAwareAttentionUNet3d(nn.Module):
    LEVEL_COUNT = 5
    AREA_ATTENTION_LEVEL = 3
    GLOBAL_ATTENTION_LEVEL = 4
    CHECKPOINTED_LEVELS = (0, 1)
    DEEP_SUPERVISION_LEVELS = (1, 2)

    def __init__(self, architecture: ModelArchitecture):
        super().__init__()
        self.architecture = architecture
        channel_counts = list(architecture.encoder_channel_counts)
        drop_path_schedule = DropPathRateSchedule(architecture.drop_path_rate,
                                                  sum(architecture.stage_depths) + 2 * architecture.attention_block_count)
        encoder_stages: list[nn.Module] = [nn.Sequential(
            ModalityAwareStem(architecture, channel_counts[0]),
            CrossStagePartialBlock(channel_counts[0], channel_counts[0], drop_path_schedule.get_next_rates(architecture.stage_depths[0])),
        )]
        for level in range(1, self.LEVEL_COUNT):
            encoder_stages.append(nn.Sequential(*self.get_encoder_stage_layers(level, channel_counts, drop_path_schedule)))
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.decoder_stages = nn.ModuleDict({
            str(level): DecoderStage(channel_counts[level + 1], channel_counts[level], architecture) for level in range(self.LEVEL_COUNT - 1)
        })
        self.output_head = nn.Conv3d(channel_counts[0], architecture.output_channel_count, kernel_size=1)
        self.deep_supervision_heads = nn.ModuleDict({
            str(level): nn.Conv3d(channel_counts[level], architecture.output_channel_count, kernel_size=1)
            for level in self.DEEP_SUPERVISION_LEVELS
        })

    def get_attention_blocks(self, channel_count: int, area_count: int, drop_path_schedule: DropPathRateSchedule) -> list[nn.Module]:
        drop_path_rates = drop_path_schedule.get_next_rates(self.architecture.attention_block_count)
        return [AreaAttentionBlock(channel_count, self.architecture.attention_head_dimension, area_count, block_index % SPATIAL_AXIS_COUNT, rate)
                for block_index, rate in enumerate(drop_path_rates)]

    def get_encoder_stage_layers(self, level: int, channel_counts: list[int], drop_path_schedule: DropPathRateSchedule) -> list[nn.Module]:
        stage_layers: list[nn.Module] = [
            ResidualDownsample(channel_counts[level - 1], channel_counts[level]),
            CrossStagePartialBlock(channel_counts[level], channel_counts[level], drop_path_schedule.get_next_rates(self.architecture.stage_depths[level])),
        ]
        if level == self.AREA_ATTENTION_LEVEL:
            stage_layers.extend(self.get_attention_blocks(channel_counts[level], self.architecture.area_count, drop_path_schedule))
        if level == self.GLOBAL_ATTENTION_LEVEL:
            stage_layers.append(SpatialPyramidPoolingFast(channel_counts[level]))
            stage_layers.extend(self.get_attention_blocks(channel_counts[level], 1, drop_path_schedule))
        return stage_layers

    def is_checkpointing_active(self, level: int) -> bool:
        return (self.architecture.is_gradient_checkpointing_enabled and self.training and torch.is_grad_enabled()
                and level in self.CHECKPOINTED_LEVELS)

    def run_encoder_stage(self, level: int, features: torch.Tensor) -> torch.Tensor:
        encoder_stage = self.encoder_stages[level]
        if self.is_checkpointing_active(level):
            return cast(torch.Tensor, checkpoint(encoder_stage, features, use_reentrant=False))
        return encoder_stage(features)

    def run_decoder_stage(self, level: int, features: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        decoder_stage = self.decoder_stages[str(level)]
        if self.is_checkpointing_active(level):
            return cast(torch.Tensor, checkpoint(decoder_stage, features, skip_features, use_reentrant=False))
        return decoder_stage(features, skip_features)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        skip_features: list[torch.Tensor] = []
        features = image
        for level in range(self.LEVEL_COUNT):
            features = self.run_encoder_stage(level, features)
            skip_features.append(features)

        auxiliary_logits: list[torch.Tensor] = []
        for level in reversed(range(self.LEVEL_COUNT - 1)):
            features = self.run_decoder_stage(level, features, skip_features[level])
            if self.training and self.architecture.is_deep_supervision_enabled and level in self.DEEP_SUPERVISION_LEVELS:
                auxiliary_logits.append(self.deep_supervision_heads[str(level)](features))

        logits = self.output_head(features)
        return [logits, *reversed(auxiliary_logits)] if self.training else logits


SegmentationModel = ModalityAwareAttentionUNet3d


def create_model(architecture: ModelArchitecture) -> SegmentationModel:
    return ModalityAwareAttentionUNet3d(architecture)


def get_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def get_logits(model_output: ModelOutput) -> torch.Tensor:
    return model_output[0] if isinstance(model_output, list) else model_output


def save_model_checkpoint(file_path: Path, model: SegmentationModel, settings: SegmentationSettings,
                          epoch: int, metrics: dict[str, float]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_suffix(".tmp")
    torch.save({
        "model_name": MODEL_NAME,
        "model_state": model.state_dict(),
        "architecture": asdict(model.architecture),
        "preprocessing": {
            "modality_suffixes": list(MODALITY_SUFFIXES),
            "cropped_volume_shape": list(settings.cropped_volume_shape),
            "patch_shape": list(settings.patch_shape),
            "prediction_threshold": settings.prediction_threshold,
        },
        "epoch": epoch,
        "metrics": metrics,
    }, temporary_path)
    temporary_path.replace(file_path)


def load_model_checkpoint(file_path: Path, device: torch.device) -> tuple[SegmentationModel, dict[str, Any]]:
    checkpoint_content = torch.load(file_path, map_location=device, weights_only=False)
    model = create_model(ModelArchitecture.from_dictionary(checkpoint_content["architecture"]))
    model.load_state_dict(checkpoint_content["model_state"])
    model.to(device).eval()
    checkpoint_metadata = {key: value for key, value in checkpoint_content.items() if key != "model_state"}
    return model, checkpoint_metadata
