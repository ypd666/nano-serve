"""Configuration dataclasses.

The first implementation keeps configuration in simple dataclasses so every
feature is explicit and easy to serialize into benchmark artifacts.
"""

from __future__ import annotations

import os
from enum import Enum
from dataclasses import asdict, dataclass, field
from typing import Literal, cast

from nano_serve.assets import DATASET_PATH_ENV, DEFAULT_MODEL_ID, MODEL_PATH_ENV
from nano_serve.scheduler.policies import SchedulerPolicy


SchedulerKind = Literal["single", "static_batch", "continuous", "chunked_prefill"]
KVCacheKind = Literal["none", "contiguous", "paged", "paged_prefix", "offload"]
AttentionBackendKind = Literal[
    "torch_naive",
    "torch_sdpa",
    "torch_gather_paged",
    "tile_paged",
]
SamplerKind = Literal["greedy", "topk_topp", "beam"]
SpecDecodeKind = Literal["none", "draft_model", "ngram", "medusa", "eagle"]
GraphKind = Literal["none", "torch_compile", "cuda_graph"]
ParallelMode = Literal["none", "dp", "tp", "pp", "ep", "pd", "af"]
WeightQuantizationKind = Literal["none", "int8", "int4"]
KVQuantizationKind = Literal["none", "int8", "fp8"]
StructuredOutputKind = Literal["none", "json_object"]


@dataclass(frozen=True)
class ParallelConfig:
    mode: ParallelMode = "none"
    tp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1


@dataclass(frozen=True)
class BenchmarkConfig:
    enable_nvtx: bool = True
    enable_ncu: bool = False
    log_iteration_trace: bool = True


@dataclass(frozen=True)
class AdvancedFeatureConfig:
    weight_quantization: WeightQuantizationKind = "none"
    kv_quantization: KVQuantizationKind = "none"
    lora: bool = False
    structured_output: StructuredOutputKind = "none"


@dataclass(frozen=True)
class EngineConfig:
    model_id: str = DEFAULT_MODEL_ID
    model_path: str | None = field(default_factory=lambda: os.environ.get(MODEL_PATH_ENV))
    dataset_path: str | None = field(
        default_factory=lambda: os.environ.get(DATASET_PATH_ENV)
    )
    scheduler: SchedulerKind = "single"
    scheduler_policy: SchedulerPolicy = SchedulerPolicy.FCFS
    kv_cache: KVCacheKind = "none"
    attention_backend: AttentionBackendKind = "torch_naive"
    sampler: SamplerKind = "greedy"
    spec_decode: SpecDecodeKind = "none"
    graph: GraphKind = "none"
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 4096
    max_prefill_chunk_tokens: int = 1024
    block_size: int = 16
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    advanced: AdvancedFeatureConfig = field(default_factory=AdvancedFeatureConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _jsonable(asdict(self)))


def _jsonable(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
