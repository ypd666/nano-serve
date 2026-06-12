"""Model and benchmark dataset asset helpers.

The first implementation target is intentionally narrow: Qwen3.5-4B plus a
ShareGPT-style serving benchmark dataset. Local asset paths are controlled by
environment variables so large files never need to live in git.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


MODEL_PATH_ENV = "NANO_SERVE_MODEL_PATH"
DATASET_PATH_ENV = "NANO_SERVE_DATASET_PATH"
MODEL_ID_ENV = "NANO_SERVE_MODEL_ID"
DATASET_REPO_ID_ENV = "NANO_SERVE_DATASET_REPO_ID"
DATASET_FILENAME_ENV = "NANO_SERVE_DATASET_FILENAME"

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_DATASET_REPO_ID = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DEFAULT_DATASET_FILENAME = "ShareGPT_V3_unfiltered_cleaned_split.json"


AssetKind = Literal["model", "dataset"]


@dataclass(frozen=True)
class AssetConfig:
    model_path: Path
    dataset_path: Path
    model_id: str = DEFAULT_MODEL_ID
    dataset_repo_id: str = DEFAULT_DATASET_REPO_ID
    dataset_filename: str = DEFAULT_DATASET_FILENAME

    @classmethod
    def from_env(cls) -> "AssetConfig":
        return cls(
            model_path=_required_path_env(MODEL_PATH_ENV),
            dataset_path=_required_path_env(DATASET_PATH_ENV),
            model_id=os.environ.get(MODEL_ID_ENV, DEFAULT_MODEL_ID),
            dataset_repo_id=os.environ.get(
                DATASET_REPO_ID_ENV,
                DEFAULT_DATASET_REPO_ID,
            ),
            dataset_filename=os.environ.get(
                DATASET_FILENAME_ENV,
                DEFAULT_DATASET_FILENAME,
            ),
        )


def env_template() -> str:
    return "\n".join(
        [
            f"export {MODEL_PATH_ENV}=$PWD/.nano-serve/models/qwen3.5-4b",
            (
                f"export {DATASET_PATH_ENV}="
                "$PWD/.nano-serve/datasets/sharegpt/"
                f"{DEFAULT_DATASET_FILENAME}"
            ),
            f"export {MODEL_ID_ENV}={DEFAULT_MODEL_ID}",
            f"export {DATASET_REPO_ID_ENV}={DEFAULT_DATASET_REPO_ID}",
            f"export {DATASET_FILENAME_ENV}={DEFAULT_DATASET_FILENAME}",
        ]
    )


def download_assets_from_env(
    *,
    force: bool = False,
    model: bool = True,
    dataset: bool = True,
    check_gitignore: bool = True,
) -> AssetConfig:
    config = AssetConfig.from_env()
    if check_gitignore:
        ensure_asset_paths_gitignored(config)
    if model:
        download_model(config, force=force)
    if dataset:
        download_serving_dataset(config, force=force)
    return config


def download_model(config: AssetConfig, *, force: bool = False) -> Path:
    if _has_files(config.model_path) and not force:
        return config.model_path

    hub = _import_huggingface_hub()
    config.model_path.mkdir(parents=True, exist_ok=True)
    hub.snapshot_download(
        repo_id=config.model_id,
        repo_type="model",
        local_dir=str(config.model_path),
    )
    return config.model_path


def download_serving_dataset(config: AssetConfig, *, force: bool = False) -> Path:
    if config.dataset_path.exists() and config.dataset_path.stat().st_size > 0 and not force:
        return config.dataset_path

    hub = _import_huggingface_hub()
    config.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hub.hf_hub_download(
            repo_id=config.dataset_repo_id,
            repo_type="dataset",
            filename=config.dataset_filename,
            local_dir=str(config.dataset_path.parent),
            force_download=force,
        )
    )

    if downloaded.resolve() != config.dataset_path.resolve():
        if config.dataset_path.exists():
            config.dataset_path.unlink()
        try:
            os.link(downloaded, config.dataset_path)
        except OSError:
            shutil.copyfile(downloaded, config.dataset_path)

    return config.dataset_path


def ensure_asset_paths_gitignored(config: AssetConfig) -> None:
    for kind, path in (("model", config.model_path), ("dataset", config.dataset_path)):
        ensure_gitignored(path, kind=kind)


def ensure_gitignored(path: Path, *, kind: AssetKind) -> None:
    repo_root = _git_repo_root()
    if repo_root is None:
        return
    path = _resolve_non_strict(path)
    if not path.is_relative_to(repo_root):
        return

    relpath = path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", str(relpath)],
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{kind} path is inside the repository but is not gitignored: {relpath}. "
            "Use a path under .nano-serve/, models/, datasets/, or data/, or update "
            ".gitignore before downloading large assets."
        )


def _required_path_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required. Example:\n\n{env_template()}"
        )
    return _resolve_non_strict(Path(value).expanduser())


def _resolve_non_strict(path: Path) -> Path:
    return path.resolve(strict=False)


def _has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir()) if path.is_dir() else path.exists()


def _git_repo_root() -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def _import_huggingface_hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "Downloading assets requires huggingface_hub. Install it with "
            "`python3 -m pip install huggingface_hub`, or install the project "
            "with the assets extra once packaging is set up: `pip install -e .[assets]`."
        ) from exc
    return huggingface_hub

