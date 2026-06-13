#!/usr/bin/env python3
"""Run Phase 7 TileLang benchmark on an explicit remote host.

The script intentionally requires `--host`; it does not inspect local SSH
configuration or infer private host aliases.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH target, for example user@h100")
    parser.add_argument(
        "--remote-dir",
        default="~/nano-serve",
        help="remote checkout path",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/ypd666/nano-serve.git",
        help="repository URL to clone when remote-dir does not exist",
    )
    parser.add_argument("--branch", default="main")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--query-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/phase7-h100")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the remote command without executing ssh",
    )
    args = parser.parse_args(argv)

    remote_command = _remote_command(args)
    ssh_command = ["ssh", args.host, remote_command]
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in ssh_command))
        return 0

    result = subprocess.run(ssh_command, check=False)
    return int(result.returncode)


def _remote_command(args: argparse.Namespace) -> str:
    remote_dir = args.remote_dir
    repo_url = shlex.quote(args.repo_url)
    branch = shlex.quote(args.branch)
    output_dir = shlex.quote(args.output_dir)
    commands = [
        "set -euo pipefail",
        (
            f"if [ ! -d {remote_dir}/.git ]; then "
            f"git clone --branch {branch} {repo_url} {remote_dir}; "
            f"else git -C {remote_dir} fetch origin {branch} "
            f"&& git -C {remote_dir} checkout {branch} "
            f"&& git -C {remote_dir} pull --ff-only origin {branch}; fi"
        ),
        f"cd {remote_dir}",
        "uv sync --extra torch --extra tilelang --extra dev",
        (
            "uv run python -m nano_serve.cli phase7-kernels --require-tilelang "
            f"--output-dir {output_dir} "
            f"--hidden-size {args.hidden_size} "
            f"--seq-len {args.seq_len} "
            f"--batch-size {args.batch_size} "
            f"--query-heads {args.query_heads} "
            f"--kv-heads {args.kv_heads} "
            f"--head-dim {args.head_dim} "
            f"--context-len {args.context_len} "
            f"--block-size {args.block_size} "
            f"--repeats {args.repeats}"
        ),
    ]
    return " && ".join(f"({command})" for command in commands)


if __name__ == "__main__":
    raise SystemExit(main())
