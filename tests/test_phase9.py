from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.phase9 import (
    Phase9PrefixCacheBenchmarkConfig,
    run_phase9_prefix_cache_benchmark,
)


def test_phase9_prefix_cache_benchmark_records_hits_and_events(tmp_path: Path) -> None:
    summary = run_phase9_prefix_cache_benchmark(
        Phase9PrefixCacheBenchmarkConfig(
            output_dir=tmp_path,
            requests=4,
            shared_prefix_tokens=8,
            unique_suffix_tokens=4,
            block_size=4,
            cache_blocks=64,
        )
    )

    assert summary["phase"] == "phase9"
    assert summary["status"] == "ok"
    assert summary["candidate"]["saved_prefill_tokens"] > 0
    assert summary["candidate"]["used_blocks"] < summary["baseline"]["used_blocks"]
    assert summary["candidate"]["prefix_hit_rate"] > 0

    event_path = Path(summary["artifacts"]["events"])
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    names = {event["name"] for event in events}
    assert "prefix_cache_lookup" in names
    assert "prefix_cache_insert" in names
    assert "prefix_cache_request_end" in names
    assert "prefix_cache_case" in names
