"""Central filesystem locations for the project.

Everything is resolved relative to the repository root so scripts work no matter
which directory they are launched from. Directories are created on import.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
CURL_DATA_DIR = DATA_DIR / "curl"   # X_/Y_ {past,future}_{left,right}.npy curl datasets
RAW_DATA_DIR = DATA_DIR / "raw"     # freshly collected recordings (npy / csv)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"     # figures, benchmark tables, etc.
MODELS_DIR = REPO_ROOT / "models_saved"     # trained PPO policy / value networks

for _d in (DATA_DIR, CURL_DATA_DIR, RAW_DATA_DIR, ARTIFACTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
