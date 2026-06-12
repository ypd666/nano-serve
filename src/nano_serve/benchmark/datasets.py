"""Serving benchmark dataset readers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSample:
    sample_id: str
    prompt: str
    reference_output: str
    source_index: int


@dataclass(frozen=True)
class DatasetLoadResult:
    samples: list[ServingSample]
    skipped: int
    total: int
    path: Path


def load_sharegpt_dataset(path: Path, *, max_samples: int | None = None) -> DatasetLoadResult:
    with path.open(encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, list):
        raise ValueError(f"ShareGPT dataset must be a JSON list: {path}")

    samples: list[ServingSample] = []
    skipped = 0
    for source_index, item in enumerate(raw):
        sample = _normalize_sharegpt_item(item, source_index)
        if sample is None:
            skipped += 1
            continue
        samples.append(sample)
        if max_samples is not None and len(samples) >= max_samples:
            break

    return DatasetLoadResult(samples=samples, skipped=skipped, total=len(raw), path=path)


def _normalize_sharegpt_item(item: object, source_index: int) -> ServingSample | None:
    if not isinstance(item, dict):
        return None
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return None

    prompt: str | None = None
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        speaker = str(turn.get("from", "")).lower()
        value = turn.get("value")
        if not isinstance(value, str) or not value.strip():
            continue
        if prompt is None and speaker in {"human", "user"}:
            prompt = value.strip()
            continue
        if prompt is not None and speaker in {"gpt", "assistant"}:
            sample_id = str(item.get("id", source_index))
            return ServingSample(
                sample_id=sample_id,
                prompt=prompt,
                reference_output=value.strip(),
                source_index=source_index,
            )
    return None

