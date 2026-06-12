"""Temperature, top-k, and top-p sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nano_serve.sampling.base import SamplingParams
from nano_serve.sampling.greedy import GreedySampler


@dataclass
class TopKTopPSampler:
    generator: Any | None = None

    def sample(self, logits: Any, params: SamplingParams | None = None) -> int:
        params = params or SamplingParams()
        if _is_greedy(params):
            return GreedySampler().sample(logits, params)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("TopKTopPSampler requires torch tensors.") from exc

        tensor = torch.as_tensor(logits).float()
        if tensor.ndim != 1:
            raise ValueError(f"logits must be 1D, got shape {tuple(tensor.shape)}")
        if params.temperature <= 0:
            raise ValueError("temperature must be positive for stochastic sampling")

        filtered = tensor / params.temperature
        filtered = _apply_top_k(filtered, params.top_k)
        filtered = _apply_top_p(filtered, params.top_p)
        probabilities = torch.softmax(filtered, dim=-1)
        if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
            raise ValueError("sampling probabilities are invalid after filtering")
        token = torch.multinomial(probabilities, num_samples=1, generator=self.generator)
        return int(token.item())


def _is_greedy(params: SamplingParams) -> bool:
    return params.temperature == 0 or (
        params.temperature == 1.0 and params.top_k is None and params.top_p is None
    )


def _apply_top_k(logits: Any, top_k: int | None):
    if top_k is None:
        return logits
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_k >= logits.numel():
        return logits
    values, _ = logits.topk(top_k)
    threshold = values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: Any, top_p: float | None):
    if top_p is None:
        return logits
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_p >= 1:
        return logits

    sorted_logits, sorted_indices = logits.sort(descending=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)
    remove_mask = cumulative_probs > top_p
    remove_mask[1:] = remove_mask[:-1].clone()
    remove_mask[0] = False

    filtered_sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
    filtered = logits.new_full(logits.shape, float("-inf"))
    return filtered.scatter(0, sorted_indices, filtered_sorted_logits)

