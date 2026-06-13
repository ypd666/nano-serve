"""Command line entrypoint for nano-serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_serve import __version__
from nano_serve.assets import download_assets_from_env, env_template
from nano_serve.benchmark.compare import compare_runs, render_compare_markdown
from nano_serve.benchmark.offline import OfflineBenchmarkConfig, run_offline_benchmark
from nano_serve.benchmark.phase10 import (
    Phase10OverlapGraphBenchmarkConfig,
    run_phase10_overlap_graph_benchmark,
)
from nano_serve.benchmark.phase11 import (
    Phase11SpeculativeBenchmarkConfig,
    run_phase11_speculative_benchmark,
)
from nano_serve.benchmark.phase5 import Phase5KVBenchmarkConfig, run_phase5_kv_benchmark
from nano_serve.benchmark.phase6 import (
    Phase6PagedAttentionBenchmarkConfig,
    run_phase6_paged_attention_benchmark,
)
from nano_serve.benchmark.phase7 import (
    Phase7KernelBenchmarkConfig,
    run_phase7_kernel_benchmark,
)
from nano_serve.benchmark.phase8 import (
    Phase8ChunkedPrefillBenchmarkConfig,
    run_phase8_chunked_prefill_benchmark,
)
from nano_serve.benchmark.phase9 import (
    Phase9PrefixCacheBenchmarkConfig,
    run_phase9_prefix_cache_benchmark,
)
from nano_serve.benchmark.phase0 import Phase0SmokeConfig, run_phase0_smoke
from nano_serve.engine.config import EngineConfig
from nano_serve.scheduler.policies import SchedulerPolicy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nano-serve",
        description="Learning-oriented LLM serving engine.",
    )
    parser.add_argument("--version", action="version", version=f"nano-serve {__version__}")

    subcommands = parser.add_subparsers(dest="command")

    assets = subcommands.add_parser("assets", help="Asset helper commands.")
    asset_commands = assets.add_subparsers(dest="asset_command")
    asset_env = asset_commands.add_parser(
        "env",
        help="Print the model and dataset environment variable template.",
    )
    asset_env.set_defaults(func=_assets_env)
    asset_download = asset_commands.add_parser(
        "download",
        help="Download the configured model and serving dataset assets.",
    )
    asset_download.add_argument("--model", action="store_true", help="download only model")
    asset_download.add_argument(
        "--dataset",
        action="store_true",
        help="download only dataset",
    )
    asset_download.add_argument("--force", action="store_true", help="force re-download")
    asset_download.add_argument(
        "--skip-gitignore-check",
        action="store_true",
        help="do not require repo-local asset paths to be gitignored",
    )
    asset_download.set_defaults(func=_assets_download)

    show_config = subcommands.add_parser(
        "show-config",
        help="Print the default engine config as JSON.",
    )
    show_config.set_defaults(func=_show_config)

    phase0 = subcommands.add_parser(
        "phase0-smoke",
        help="Run the Phase 0 local infrastructure smoke.",
    )
    _add_phase0_smoke_args(phase0)
    phase0.set_defaults(func=_phase0_smoke)

    phase1 = subcommands.add_parser(
        "phase1-offline",
        help="Run the Phase 1 single-request torch offline benchmark.",
    )
    _add_phase1_offline_args(phase1)
    phase1.set_defaults(func=_phase1_offline)

    phase5 = subcommands.add_parser(
        "phase5-kv",
        help="Run the Phase 5 paged KV allocator benchmark.",
    )
    _add_phase5_kv_args(phase5)
    phase5.set_defaults(func=_phase5_kv)

    phase6 = subcommands.add_parser(
        "phase6-attention",
        help="Run the Phase 6 torch gather paged-attention benchmark.",
    )
    _add_phase6_attention_args(phase6)
    phase6.set_defaults(func=_phase6_attention)

    phase7 = subcommands.add_parser(
        "phase7-kernels",
        help="Run the Phase 7 TileLang kernel benchmark harness.",
    )
    _add_phase7_kernel_args(phase7)
    phase7.set_defaults(func=_phase7_kernels)

    phase8 = subcommands.add_parser(
        "phase8-chunked-prefill",
        help="Run the Phase 8 chunked-prefill scheduler benchmark.",
    )
    _add_phase8_chunked_prefill_args(phase8)
    phase8.set_defaults(func=_phase8_chunked_prefill)

    phase9 = subcommands.add_parser(
        "phase9-prefix-cache",
        help="Run the Phase 9 prefix-cache benchmark.",
    )
    _add_phase9_prefix_cache_args(phase9)
    phase9.set_defaults(func=_phase9_prefix_cache)

    phase10 = subcommands.add_parser(
        "phase10-overlap-graphs",
        help="Run the Phase 10 overlap and graph benchmark.",
    )
    _add_phase10_overlap_graph_args(phase10)
    phase10.set_defaults(func=_phase10_overlap_graphs)

    phase11 = subcommands.add_parser(
        "phase11-speculative",
        help="Run the Phase 11 speculative decoding benchmark.",
    )
    _add_phase11_speculative_args(phase11)
    phase11.set_defaults(func=_phase11_speculative)

    bench = subcommands.add_parser("bench", help="Benchmark helper commands.")
    bench_commands = bench.add_subparsers(dest="bench_command")
    bench_dummy = bench_commands.add_parser(
        "dummy",
        help="Alias for the Phase 0 deterministic smoke benchmark.",
    )
    _add_phase0_smoke_args(bench_dummy)
    bench_dummy.set_defaults(func=_phase0_smoke)
    bench_offline = bench_commands.add_parser(
        "offline",
        help="Alias for the Phase 1 single-request torch offline benchmark.",
    )
    _add_phase1_offline_args(bench_offline)
    bench_offline.set_defaults(func=_phase1_offline)
    bench_kv = bench_commands.add_parser(
        "kv",
        help="Alias for the Phase 5 paged KV allocator benchmark.",
    )
    _add_phase5_kv_args(bench_kv)
    bench_kv.set_defaults(func=_phase5_kv)
    bench_attention = bench_commands.add_parser(
        "attention",
        help="Alias for the Phase 6 torch gather paged-attention benchmark.",
    )
    _add_phase6_attention_args(bench_attention)
    bench_attention.set_defaults(func=_phase6_attention)
    bench_kernels = bench_commands.add_parser(
        "kernels",
        help="Alias for the Phase 7 TileLang kernel benchmark harness.",
    )
    _add_phase7_kernel_args(bench_kernels)
    bench_kernels.set_defaults(func=_phase7_kernels)
    bench_chunked_prefill = bench_commands.add_parser(
        "chunked-prefill",
        help="Alias for the Phase 8 chunked-prefill scheduler benchmark.",
    )
    _add_phase8_chunked_prefill_args(bench_chunked_prefill)
    bench_chunked_prefill.set_defaults(func=_phase8_chunked_prefill)
    bench_prefix_cache = bench_commands.add_parser(
        "prefix-cache",
        help="Alias for the Phase 9 prefix-cache benchmark.",
    )
    _add_phase9_prefix_cache_args(bench_prefix_cache)
    bench_prefix_cache.set_defaults(func=_phase9_prefix_cache)
    bench_overlap_graphs = bench_commands.add_parser(
        "overlap-graphs",
        help="Alias for the Phase 10 overlap and graph benchmark.",
    )
    _add_phase10_overlap_graph_args(bench_overlap_graphs)
    bench_overlap_graphs.set_defaults(func=_phase10_overlap_graphs)
    bench_speculative = bench_commands.add_parser(
        "speculative",
        help="Alias for the Phase 11 speculative decoding benchmark.",
    )
    _add_phase11_speculative_args(bench_speculative)
    bench_speculative.set_defaults(func=_phase11_speculative)
    bench_compare = bench_commands.add_parser(
        "compare",
        help="Compare two benchmark run summaries or run directories.",
    )
    bench_compare.add_argument("base", type=Path, help="base run dir or summary.json")
    bench_compare.add_argument(
        "candidate",
        type=Path,
        help="candidate run dir or summary.json",
    )
    bench_compare.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output format",
    )
    bench_compare.set_defaults(func=_bench_compare)

    serve = subcommands.add_parser("serve", help="Server mode is not implemented yet.")
    serve.set_defaults(func=_not_implemented)

    return parser


def _add_phase0_smoke_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase0"),
        help="directory for run artifacts",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="number of ShareGPT samples to load",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="opt into the heavy full model load path",
    )


def _add_phase1_offline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase1"),
        help="directory for run artifacts",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="number of ShareGPT samples to generate",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8,
        help="maximum generated tokens per request",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=128,
        help="truncate prompts to this many tokens",
    )
    parser.add_argument(
        "--workload",
        default="single_short",
        help="workload name recorded in artifacts",
    )
    parser.add_argument(
        "--kv-cache",
        choices=("none", "contiguous"),
        default="none",
        help="KV cache backend for the torch offline benchmark",
    )
    parser.add_argument(
        "--scheduler",
        choices=("single", "static_batch", "continuous", "chunked_prefill"),
        default="single",
        help="scheduler mode for the torch offline benchmark",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=tuple(policy.value for policy in SchedulerPolicy),
        default=SchedulerPolicy.FCFS.value,
        help="continuous scheduler policy",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="max active sequences for static/continuous batching",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="maximum full-context tokens selected per continuous iteration",
    )
    parser.add_argument(
        "--max-prefill-chunk-tokens",
        type=int,
        default=1024,
        help="maximum prompt tokens selected per chunked-prefill request",
    )


def _add_phase5_kv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase5"),
        help="directory for run artifacts",
    )
    parser.add_argument("--num-blocks", type=int, default=128, help="number of KV blocks")
    parser.add_argument("--block-size", type=int, default=16, help="tokens per KV block")
    parser.add_argument("--num-requests", type=int, default=64, help="requests to simulate")
    parser.add_argument(
        "--max-prefill-tokens",
        type=int,
        default=128,
        help="maximum prefill length per request",
    )
    parser.add_argument(
        "--max-decode-tokens",
        type=int,
        default=64,
        help="maximum decode append tokens per request",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")


def _add_phase6_attention_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase6"),
        help="directory for run artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="query batch size")
    parser.add_argument("--query-heads", type=int, default=8, help="number of query heads")
    parser.add_argument("--kv-heads", type=int, default=2, help="number of KV heads")
    parser.add_argument("--head-dim", type=int, default=64, help="attention head dimension")
    parser.add_argument(
        "--context-lens",
        default="128,512,1024",
        help="comma-separated context lengths to sweep",
    )
    parser.add_argument(
        "--block-sizes",
        default="8,16,32",
        help="comma-separated paged KV block sizes to sweep",
    )
    parser.add_argument("--repeats", type=int, default=5, help="repeats per sweep case")
    parser.add_argument("--seed", type=int, default=0, help="random seed")


def _add_phase7_kernel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase7"),
        help="directory for run artifacts",
    )
    parser.add_argument("--hidden-size", type=int, default=512, help="operator hidden size")
    parser.add_argument("--seq-len", type=int, default=128, help="operator sequence length")
    parser.add_argument("--batch-size", type=int, default=2, help="operator batch size")
    parser.add_argument("--query-heads", type=int, default=8, help="number of query heads")
    parser.add_argument("--kv-heads", type=int, default=2, help="number of KV heads")
    parser.add_argument("--head-dim", type=int, default=64, help="attention head dimension")
    parser.add_argument("--context-len", type=int, default=512, help="paged attention context")
    parser.add_argument("--block-size", type=int, default=16, help="paged KV block size")
    parser.add_argument("--repeats", type=int, default=10, help="repeats per kernel case")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--require-tilelang",
        action="store_true",
        help="skip instead of using torch fallback when TileLang is unavailable",
    )
    parser.add_argument(
        "--enable-ncu",
        action="store_true",
        help="record Nsight Compute profiling intent in benchmark artifacts",
    )


def _add_phase8_chunked_prefill_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase8"),
        help="directory for run artifacts",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="128,512,2048",
        help="comma-separated max prefill chunk sizes to sweep",
    )
    parser.add_argument(
        "--long-prompt-tokens",
        type=int,
        default=8192,
        help="long prompt length in simulated tokens",
    )
    parser.add_argument(
        "--decode-requests",
        type=int,
        default=8,
        help="number of running decode requests",
    )
    parser.add_argument(
        "--decode-tokens-per-request",
        type=int,
        default=128,
        help="decode tokens generated by each running request",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=9,
        help="maximum active sequences",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="maximum tokens selected per scheduler iteration",
    )
    parser.add_argument(
        "--prefill-token-time-ms",
        type=float,
        default=0.02,
        help="simulated cost per prefill token",
    )
    parser.add_argument(
        "--decode-token-time-ms",
        type=float,
        default=0.05,
        help="simulated cost per decode token",
    )


def _add_phase9_prefix_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase9"),
        help="directory for run artifacts",
    )
    parser.add_argument("--requests", type=int, default=64, help="requests to simulate")
    parser.add_argument(
        "--shared-prefix-tokens",
        type=int,
        default=512,
        help="shared prompt prefix length in tokens",
    )
    parser.add_argument(
        "--unique-suffix-tokens",
        type=int,
        default=64,
        help="per-request unique suffix length in tokens",
    )
    parser.add_argument("--block-size", type=int, default=16, help="KV block size")
    parser.add_argument("--cache-blocks", type=int, default=4096, help="KV block capacity")
    parser.add_argument(
        "--max-prefix-entries",
        type=int,
        default=None,
        help="maximum cached prefix entries before LRU eviction",
    )
    parser.add_argument(
        "--prefill-token-time-ms",
        type=float,
        default=0.02,
        help="simulated cost per prefill token",
    )


def _add_phase10_overlap_graph_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase10"),
        help="directory for run artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="decode batch size")
    parser.add_argument("--hidden-size", type=int, default=512, help="hidden dimension")
    parser.add_argument("--decode-steps", type=int, default=256, help="repeated decode steps")
    parser.add_argument(
        "--bucket-batch-sizes",
        default="1,2,4,8",
        help="comma-separated graph bucket batch sizes",
    )
    parser.add_argument(
        "--bucket-seq-lens",
        default="1,2,4,8",
        help="comma-separated graph bucket sequence lengths",
    )
    parser.add_argument(
        "--disable-torch-compile",
        action="store_true",
        help="skip the torch.compile experiment",
    )
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="skip the CUDA graph experiment",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")


def _add_phase11_speculative_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/phase11"),
        help="directory for run artifacts",
    )
    parser.add_argument(
        "--gamma-values",
        default="1,2,4,8",
        help="comma-separated speculative gamma values to sweep",
    )
    parser.add_argument("--output-tokens", type=int, default=256, help="tokens per request")
    parser.add_argument("--batch-size", type=int, default=4, help="requests per case")
    parser.add_argument("--prompt-tokens", type=int, default=16, help="prompt tokens")
    parser.add_argument(
        "--target-step-time-ms",
        type=float,
        default=1.0,
        help="simulated target model time per verification call",
    )
    parser.add_argument(
        "--draft-token-time-ms",
        type=float,
        default=0.1,
        help="simulated draft model time per proposed token",
    )


def _assets_env(_: argparse.Namespace) -> int:
    print(env_template())
    return 0


def _assets_download(args: argparse.Namespace) -> int:
    download_model = args.model or not args.dataset
    download_dataset = args.dataset or not args.model
    config = download_assets_from_env(
        force=args.force,
        model=download_model,
        dataset=download_dataset,
        check_gitignore=not args.skip_gitignore_check,
    )
    if download_model:
        print(f"model: {config.model_id} -> {config.model_path}")
    if download_dataset:
        print(
            "dataset: "
            f"{config.dataset_repo_id}/{config.dataset_filename} -> "
            f"{config.dataset_path}"
        )
    return 0


def _show_config(_: argparse.Namespace) -> int:
    print(json.dumps(EngineConfig().to_dict(), indent=2, sort_keys=True))
    return 0


def _phase0_smoke(args: argparse.Namespace) -> int:
    summary = run_phase0_smoke(
        Phase0SmokeConfig(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            load_model=args.load_model,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase1_offline(args: argparse.Namespace) -> int:
    summary = run_offline_benchmark(
        OfflineBenchmarkConfig(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            workload=args.workload,
            kv_cache=args.kv_cache,
            scheduler=args.scheduler,
            scheduler_policy=SchedulerPolicy(args.scheduler_policy),
            batch_size=args.batch_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase5_kv(args: argparse.Namespace) -> int:
    summary = run_phase5_kv_benchmark(
        Phase5KVBenchmarkConfig(
            output_dir=args.output_dir,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            num_requests=args.num_requests,
            max_prefill_tokens=args.max_prefill_tokens,
            max_decode_tokens=args.max_decode_tokens,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase6_attention(args: argparse.Namespace) -> int:
    summary = run_phase6_paged_attention_benchmark(
        Phase6PagedAttentionBenchmarkConfig(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            context_lens=_parse_int_list(args.context_lens, name="context_lens"),
            block_sizes=_parse_int_list(args.block_sizes, name="block_sizes"),
            repeats=args.repeats,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase7_kernels(args: argparse.Namespace) -> int:
    summary = run_phase7_kernel_benchmark(
        Phase7KernelBenchmarkConfig(
            output_dir=args.output_dir,
            hidden_size=args.hidden_size,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            query_heads=args.query_heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            context_len=args.context_len,
            block_size=args.block_size,
            repeats=args.repeats,
            seed=args.seed,
            require_tilelang=args.require_tilelang,
            enable_ncu=args.enable_ncu,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase8_chunked_prefill(args: argparse.Namespace) -> int:
    summary = run_phase8_chunked_prefill_benchmark(
        Phase8ChunkedPrefillBenchmarkConfig(
            output_dir=args.output_dir,
            chunk_sizes=_parse_int_list(args.chunk_sizes, name="chunk-sizes"),
            long_prompt_tokens=args.long_prompt_tokens,
            decode_requests=args.decode_requests,
            decode_tokens_per_request=args.decode_tokens_per_request,
            max_num_seqs=args.batch_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            prefill_token_time_ms=args.prefill_token_time_ms,
            decode_token_time_ms=args.decode_token_time_ms,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase9_prefix_cache(args: argparse.Namespace) -> int:
    summary = run_phase9_prefix_cache_benchmark(
        Phase9PrefixCacheBenchmarkConfig(
            output_dir=args.output_dir,
            requests=args.requests,
            shared_prefix_tokens=args.shared_prefix_tokens,
            unique_suffix_tokens=args.unique_suffix_tokens,
            block_size=args.block_size,
            cache_blocks=args.cache_blocks,
            max_prefix_entries=args.max_prefix_entries,
            prefill_token_time_ms=args.prefill_token_time_ms,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase10_overlap_graphs(args: argparse.Namespace) -> int:
    summary = run_phase10_overlap_graph_benchmark(
        Phase10OverlapGraphBenchmarkConfig(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            hidden_size=args.hidden_size,
            decode_steps=args.decode_steps,
            bucket_batch_sizes=_parse_int_list(
                args.bucket_batch_sizes,
                name="bucket-batch-sizes",
            ),
            bucket_seq_lens=_parse_int_list(args.bucket_seq_lens, name="bucket-seq-lens"),
            enable_torch_compile=not args.disable_torch_compile,
            enable_cuda_graph=not args.disable_cuda_graph,
            seed=args.seed,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _phase11_speculative(args: argparse.Namespace) -> int:
    summary = run_phase11_speculative_benchmark(
        Phase11SpeculativeBenchmarkConfig(
            output_dir=args.output_dir,
            gamma_values=_parse_int_list(args.gamma_values, name="gamma-values"),
            output_tokens=args.output_tokens,
            batch_size=args.batch_size,
            prompt_tokens=args.prompt_tokens,
            target_step_time_ms=args.target_step_time_ms,
            draft_token_time_ms=args.draft_token_time_ms,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _bench_compare(args: argparse.Namespace) -> int:
    comparison = compare_runs(args.base, args.candidate)
    if args.format == "json":
        print(json.dumps(comparison, indent=2, sort_keys=True))
    else:
        print(render_compare_markdown(comparison), end="")
    return 0


def _not_implemented(_: argparse.Namespace) -> int:
    raise NotImplementedError("Server mode is not implemented yet.")


def _parse_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
