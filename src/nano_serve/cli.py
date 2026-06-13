"""Command line entrypoint for nano-serve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nano_serve import __version__
from nano_serve.assets import download_assets_from_env, env_template
from nano_serve.benchmark.compare import compare_runs, render_compare_markdown
from nano_serve.benchmark.offline import OfflineBenchmarkConfig, run_offline_benchmark
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
        choices=("single", "static_batch", "continuous"),
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
