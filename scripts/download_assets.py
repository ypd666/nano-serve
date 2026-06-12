#!/usr/bin/env python3
"""Download nano-serve's first-stage model and serving benchmark dataset."""

from __future__ import annotations

import argparse
import sys

from nano_serve.assets import download_assets_from_env, env_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="download only the model")
    parser.add_argument("--dataset", action="store_true", help="download only the dataset")
    parser.add_argument("--force", action="store_true", help="force re-download")
    parser.add_argument(
        "--skip-gitignore-check",
        action="store_true",
        help="do not require repo-local asset paths to be gitignored",
    )
    parser.add_argument(
        "--print-env-template",
        action="store_true",
        help="print the expected environment variables and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.print_env_template:
        print(env_template())
        return 0

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


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

