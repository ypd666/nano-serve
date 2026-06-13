from __future__ import annotations

import os
from pathlib import Path

from nano_serve import Engine, EngineConfig, FeatureFlags, SamplingParams
from nano_serve.assets import (
    DATASET_PATH_ENV,
    DEFAULT_MODEL_ID,
    MODEL_PATH_ENV,
    AssetConfig,
    env_template,
    load_dotenv,
)
from nano_serve.kv_cache import ContiguousKVCache, ContiguousKVCacheConfig, ContiguousLayerState
from nano_serve.kv_cache.paged import PagedKVCache
from nano_serve.observability import platform_event
from nano_serve.platform import detect_platform
from nano_serve.sampling.greedy import GreedySampler
from nano_serve.sampling.topk_topp import TopKTopPSampler


def test_public_imports() -> None:
    config = EngineConfig()
    flags = FeatureFlags.from_config(config)
    engine = Engine(config)

    assert config.scheduler == "single"
    assert flags.use_kv_cache is False
    assert engine.config is config


def test_greedy_sampler() -> None:
    token_id = GreedySampler().sample([0.1, 3.0, 2.0], SamplingParams())

    assert token_id == 1


def test_top_k_top_p_sampler_masks_candidates() -> None:
    import torch

    generator = torch.Generator().manual_seed(0)
    sampler = TopKTopPSampler(generator=generator)

    token_id = sampler.sample(
        torch.tensor([10.0, 9.0, 0.0]),
        SamplingParams(temperature=1.0, top_k=1),
    )

    assert token_id == 0


def test_top_p_sampler_keeps_probability_prefix() -> None:
    import torch

    generator = torch.Generator().manual_seed(0)
    sampler = TopKTopPSampler(generator=generator)

    token_id = sampler.sample(
        torch.tensor([10.0, 9.0, 1.0]),
        SamplingParams(temperature=1.0, top_p=0.8),
    )

    assert token_id in {0, 1}


def test_paged_kv_allocator_smoke() -> None:
    cache = PagedKVCache(num_blocks=4, block_size=2)
    handle = cache.allocate_prefill("req", 3)

    assert handle.block_ids
    assert cache.get_block_table("req") == handle.block_ids

    cache.allocate_decode_slot("req")
    cache.free("req")

    assert cache.get_block_table("req") == []


def test_contiguous_kv_cache_tracks_sequence_and_bytes() -> None:
    import torch

    cache = ContiguousKVCache(ContiguousKVCacheConfig(max_model_len=8, num_layers=1, block_size=4))
    layer_state = ContiguousLayerState(
        layer_type="full_attention",
        key=torch.zeros(1, 1, 3, 2),
        value=torch.zeros(1, 1, 3, 2),
    )

    handle = cache.allocate_prefill("req", 3, max_decode_tokens=1, layer_states=[layer_state])
    cache.allocate_decode_slot("req")

    assert handle.num_tokens == 4
    assert cache.sequence_length("req") == 4
    assert cache.get_block_table("req") == [0]
    assert cache.stats().bytes_used > 0

    cache.free("req")

    assert cache.stats().tokens_used == 0


def test_every_feature_doc_exists() -> None:
    feature_dir = Path(__file__).resolve().parents[1] / "docs" / "features"
    expected = {
        "00-infrastructure.md",
        "01-torch-forwarding.md",
        "02-tokenizer-sampling-streaming.md",
        "03-kv-cache-prefill-decode.md",
        "04-static-batching.md",
        "05-continuous-batching.md",
        "06-paged-kv-cache.md",
        "07-paged-attention-reference.md",
        "08-tilelang-kernels.md",
        "09-chunked-prefill.md",
        "10-prefix-cache-radix.md",
        "11-overlap-graphs.md",
        "12-speculative-decoding.md",
        "13-quantization-lora-structured.md",
        "14-single-node-distributed.md",
        "15-multi-node-distributed.md",
        "16-pd-disaggregation.md",
        "17-af-disaggregation.md",
        "18-observability-production.md",
    }

    assert expected.issubset({path.name for path in feature_dir.glob("*.md")})


def test_bilingual_readmes_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "README.md").exists()
    assert (root / "README.zh.md").exists()
    assert "README.zh.md" in (root / "README.md").read_text(encoding="utf-8")
    assert "README.md" in (root / "README.zh.md").read_text(encoding="utf-8")


def test_asset_env_template_mentions_required_paths() -> None:
    template = env_template()

    assert MODEL_PATH_ENV in template
    assert DATASET_PATH_ENV in template
    assert DEFAULT_MODEL_ID in template


def test_asset_config_and_engine_config_read_env(monkeypatch) -> None:
    monkeypatch.setenv(MODEL_PATH_ENV, "/tmp/nano-serve/model")
    monkeypatch.setenv(DATASET_PATH_ENV, "/tmp/nano-serve/sharegpt.json")

    asset_config = AssetConfig.from_env()
    engine_config = EngineConfig()

    assert asset_config.model_path == Path("/tmp/nano-serve/model").resolve(strict=False)
    assert asset_config.dataset_path == Path("/tmp/nano-serve/sharegpt.json").resolve(
        strict=False
    )
    assert engine_config.model_id == DEFAULT_MODEL_ID
    assert engine_config.model_path == "/tmp/nano-serve/model"
    assert engine_config.dataset_path == "/tmp/nano-serve/sharegpt.json"


def test_load_dotenv_sets_missing_values(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "\n".join(
            [
                "NANO_SERVE_MODEL_PATH=relative/model",
                "NANO_SERVE_DATASET_PATH='relative/sharegpt.json'",
                "NANO_SERVE_MODEL_ID=ignored",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv(MODEL_PATH_ENV, raising=False)
    monkeypatch.delenv(DATASET_PATH_ENV, raising=False)
    monkeypatch.setenv("NANO_SERVE_MODEL_ID", "already-set")

    load_dotenv(dotenv)

    assert os.environ[MODEL_PATH_ENV] == "relative/model"
    assert os.environ[DATASET_PATH_ENV] == "relative/sharegpt.json"
    assert os.environ["NANO_SERVE_MODEL_ID"] == "already-set"


def test_platform_detection_without_torch() -> None:
    info = detect_platform(torch_module=None)

    assert info.device_backend in {"cpu", "cuda"}
    assert "os_name" in info.to_log_fields()


def test_platform_detection_with_cuda() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 2

        @staticmethod
        def get_device_name(index: int) -> str:
            return ["NVIDIA H20", "NVIDIA H100"][index]

    class FakeTorch:
        __version__ = "2.test"
        cuda = FakeCuda()

    info = detect_platform(torch_module=FakeTorch())

    assert info.torch_version == "2.test"
    assert info.device_backend == "cuda"
    assert info.cuda_device_count == 2
    assert info.cuda_device_names == ("NVIDIA H20", "NVIDIA H100")


def test_platform_detection_without_cuda_uses_cpu() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        __version__ = "2.test"
        cuda = FakeCuda()

    info = detect_platform(torch_module=FakeTorch())

    assert info.device_backend == "cpu"
    assert info.cuda_available is False
    assert info.cuda_device_count == 0


def test_platform_event_contains_phase0_log_fields() -> None:
    event = platform_event()

    assert event.name == "platform_detected"
    assert "os_name" in event.fields
    assert "machine" in event.fields
    assert "python_version" in event.fields
    assert event.fields["device_backend"] in {"cpu", "cuda"}
