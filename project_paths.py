"""Resolve repository data paths without importing any ML packages.

TESI_PROJECT_ROOT may point to an existing Tesi-main directory on another disk.
This module performs no downloads, directory creation, training, or inference.
"""
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
_override = os.environ.get("TESI_PROJECT_ROOT")
PROJECT_ROOT = (
    Path(_override).expanduser().resolve()
    if _override else REPOSITORY_ROOT / "DATA" / "Alessandro Tesi" / "Tesi-main"
)
if not (PROJECT_ROOT / "Secondo_Esperimento").is_dir():
    raise FileNotFoundError(
        f"Missing Secondo_Esperimento under {PROJECT_ROOT}. "
        "TESI_PROJECT_ROOT must point to the Tesi-main directory."
    )
EXPERIMENT_ROOT = PROJECT_ROOT / "Secondo_Esperimento"
BASELINES_PATH = EXPERIMENT_ROOT / "Baselines"
PROCESSED_DATA_PATH = EXPERIMENT_ROOT / "Preprocessing" / "Dataset" / "ProcessedDB"
