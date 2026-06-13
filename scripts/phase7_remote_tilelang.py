#!/usr/bin/env python3
"""Run Phase 7 TileLang benchmark on an explicit remote host.

The script intentionally requires `--host`; it does not inspect local SSH
configuration or infer private host aliases.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path


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
        "--fetch-dir",
        type=Path,
        help="optional local directory for fetching the remote run artifact",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing ssh or scp",
    )
    args = parser.parse_args(argv)

    remote_command = _remote_command(args)
    ssh_command = ["ssh", args.host, remote_command]
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in ssh_command))
        if args.fetch_dir is not None:
            print(_dry_run_scp_command(args, "<remote-run-dir>"))
        return 0

    result = subprocess.run(
        ssh_command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="")
    if result.returncode != 0:
        return int(result.returncode)

    remote_run_dir = _parse_remote_run_dir(result.stdout)
    if args.fetch_dir is not None and remote_run_dir is None:
        print("remote run directory was not found in command output")
        return 1
    if args.fetch_dir is not None and remote_run_dir is not None:
        args.fetch_dir.mkdir(parents=True, exist_ok=True)
        scp_command = [
            "scp",
            "-r",
            _remote_scp_source(args.host, args.remote_dir, remote_run_dir),
            str(args.fetch_dir),
        ]
        fetch = subprocess.run(scp_command, check=False)
        if fetch.returncode != 0:
            return int(fetch.returncode)
    return 0


def _remote_command(args: argparse.Namespace) -> str:
    remote_dir = args.remote_dir
    repo_url = shlex.quote(args.repo_url)
    branch = shlex.quote(args.branch)
    output_dir = shlex.quote(args.output_dir)
    commands = [
        "set -euo pipefail",
        "export PATH=\"$HOME/.local/bin:$PATH\"",
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
            "summary_json=$(uv run python -m nano_serve.cli phase7-kernels --require-tilelang "
            f"--output-dir {output_dir} "
            f"--hidden-size {args.hidden_size} "
            f"--seq-len {args.seq_len} "
            f"--batch-size {args.batch_size} "
            f"--query-heads {args.query_heads} "
            f"--kv-heads {args.kv_heads} "
            f"--head-dim {args.head_dim} "
            f"--context-len {args.context_len} "
            f"--block-size {args.block_size} "
            f"--repeats {args.repeats}) && "
            "printf '%s\n' \"$summary_json\" && "
            "SUMMARY_JSON=\"$summary_json\" uv run python -c "
            "\"import json, os; summary=json.loads(os.environ['SUMMARY_JSON']); "
            "print('NANO_SERVE_REMOTE_RUN_DIR=' + str(summary.get('run_dir', '')))\""
        ),
    ]
    return " && ".join(commands)


def _parse_remote_run_dir(output: str) -> str | None:
    for line in reversed(output.splitlines()):
        if line.startswith("NANO_SERVE_REMOTE_RUN_DIR="):
            run_dir = line.split("=", 1)[1].strip()
            return run_dir or None
    for line in reversed(output.splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_dir = payload.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            return run_dir
    return None


def _dry_run_scp_command(args: argparse.Namespace, remote_run_dir: str) -> str:
    command = [
        "scp",
        "-r",
        _remote_scp_source(args.host, args.remote_dir, remote_run_dir),
        str(args.fetch_dir),
    ]
    return " ".join(shlex.quote(part) for part in command)


def _remote_scp_source(host: str, remote_dir: str, remote_run_dir: str) -> str:
    if remote_run_dir.startswith("/") or remote_run_dir.startswith("~"):
        artifact_path = remote_run_dir
    else:
        artifact_path = f"{remote_dir.rstrip('/')}/{remote_run_dir}"
    return f"{host}:{artifact_path}"


if __name__ == "__main__":
    raise SystemExit(main())
