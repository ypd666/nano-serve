from __future__ import annotations

import json
from pathlib import Path

from nano_serve.benchmark.phase10 import (
    Phase10OverlapGraphBenchmarkConfig,
    run_phase10_overlap_graph_benchmark,
)
from nano_serve.runtime import (
    DoubleBufferedBatchMetadata,
    ShapeBucket,
    ShapeBucketSelector,
    TokenizerWorker,
)
from nano_serve.runtime.overlap import BatchMetadata


def test_shape_bucket_selector_picks_smallest_fitting_bucket() -> None:
    selector = ShapeBucketSelector(
        [
            ShapeBucket(batch_size=8, seq_len=8),
            ShapeBucket(batch_size=4, seq_len=2),
            ShapeBucket(batch_size=4, seq_len=4),
        ]
    )

    selection = selector.select(batch_size=3, seq_len=2)

    assert selection.bucket == ShapeBucket(batch_size=4, seq_len=2)
    assert selection.padded_elements == 2


def test_double_buffered_metadata_alternates_slots() -> None:
    buffers = DoubleBufferedBatchMetadata()

    first = buffers.publish(BatchMetadata(0, ("a",), 1, 1))
    second = buffers.publish(BatchMetadata(1, ("b",), 1, 1))

    assert (first.slot, second.slot) == (0, 1)
    assert buffers.latest() is not None
    assert buffers.latest().metadata.iteration == 1


def test_tokenizer_worker_preserves_result_index() -> None:
    worker = TokenizerWorker(_ToyTokenizer(), max_workers=2)

    futures = [
        worker.submit(index=1, text="aa b"),
        worker.submit(index=0, text="c"),
    ]
    results = sorted((future.result(timeout=10) for future in futures), key=lambda item: item.index)
    worker.shutdown()

    assert [result.index for result in results] == [0, 1]
    assert [result.token_ids for result in results] == [[1], [2, 1]]


def test_phase10_benchmark_emits_graph_case_events(tmp_path: Path) -> None:
    summary = run_phase10_overlap_graph_benchmark(
        Phase10OverlapGraphBenchmarkConfig(
            output_dir=tmp_path,
            batch_size=2,
            hidden_size=16,
            decode_steps=4,
            bucket_batch_sizes=(1, 2),
            bucket_seq_lens=(1,),
            enable_torch_compile=False,
            enable_cuda_graph=False,
        )
    )

    assert summary["phase"] == "phase10"
    assert summary["status"] == "ok"
    assert summary["best_latency_case"]["name"] == "eager"
    event_path = Path(summary["artifacts"]["events"])
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    names = {event["name"] for event in events}
    assert "tokenizer_worker_task" in names
    assert "async_scheduler_prep" in names
    assert "double_buffer_publish" in names
    assert "phase10_graph_case" in names


class _ToyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(item) for item in text.split()]
