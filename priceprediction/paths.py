"""Project locations resolved independently of the process working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
REPORTS_DIR = ROOT / 'reports'
ARTIFACTS_DIR = ROOT / 'artifacts'
