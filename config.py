import os
from pathlib import Path


def get_alpaca_credentials() -> tuple[str, str]:
    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.is_file():
        with env_path.open(encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name = name.strip()
                if name in {"APCA_API_KEY_ID", "APCA_API_SECRET_KEY"}:
                    os.environ.setdefault(name, value.strip().strip("'\""))

    api_key = os.environ.get("APCA_API_KEY_ID")
    secret_key = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in .env or the environment."
        )
    return api_key, secret_key
