"""Model loader helpers for the first Qwen3.5-4B milestone."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    model_path: Path
    dtype: str = "bfloat16"
    max_model_len: int | None = None


class ModelLoader:
    def load_config(self, spec: ModelSpec) -> dict[str, Any]:
        config_path = spec.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing model config: {config_path}")

        import json

        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise ValueError(f"Model config must be a JSON object: {config_path}")
        return config

    def load(self, spec: ModelSpec):
        config = self.load_config(spec)
        if config.get("model_type") != "qwen3_5":
            raise ValueError(f"Unsupported model type: {config.get('model_type')}")

        try:
            import torch
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "ModelLoader.load requires torch and safetensors. "
                "Use the project uv environment with these packages installed."
            ) from exc

        from nano_serve.model.qwen35 import Qwen35ForCausalLM, Qwen35TextConfig

        device = _select_device(torch)
        dtype = _parse_torch_dtype(torch, spec.dtype)
        text_config = Qwen35TextConfig.from_model_config(config)
        model = Qwen35ForCausalLM(text_config, device=device, dtype=dtype)
        model.eval()

        weight_map = _load_weight_map(spec.model_path)
        parameter_sources = _parameter_sources(model)
        missing = sorted(set(parameter_sources.values()) - set(weight_map))
        if missing:
            raise KeyError(f"Missing checkpoint tensors: {missing[:8]}")

        grouped: dict[str, list[tuple[str, str]]] = {}
        for parameter_name, source_name in parameter_sources.items():
            grouped.setdefault(weight_map[source_name], []).append((parameter_name, source_name))

        named_parameters = dict(model.named_parameters())
        with torch.no_grad():
            for shard_name, names in grouped.items():
                shard_path = spec.model_path / shard_name
                with safe_open(shard_path, framework="pt", device="cpu") as shard:
                    for parameter_name, source_name in names:
                        parameter = named_parameters[parameter_name]
                        tensor = shard.get_tensor(source_name).to(
                            device=parameter.device,
                            dtype=parameter.dtype,
                        )
                        parameter.copy_(tensor)

        return model


def _load_weight_map(model_path: Path) -> dict[str, str]:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing safetensors index: {index_path}")

    import json

    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Invalid safetensors index: {index_path}")
    return {str(key): str(value) for key, value in weight_map.items()}


def _parameter_sources(model: Any) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name, parameter in model.named_parameters():
        del parameter
        if name == "lm_head.weight":
            # Qwen3.5-4B ties lm_head to token embeddings and the checkpoint does
            # not store a separate lm_head tensor.
            continue
        if not name.startswith("model."):
            raise ValueError(f"Unexpected Qwen3.5 parameter: {name}")
        sources[name] = "model.language_model." + name.removeprefix("model.")
    return sources


def _select_device(torch_module: Any):
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _parse_torch_dtype(torch_module: Any, dtype: str):
    normalized = dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype}")

