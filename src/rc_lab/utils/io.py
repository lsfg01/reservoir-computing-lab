import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Carga un archivo YAML y devuelve el dict resultante."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_result(result: Any, output_dir: str | Path) -> Path:
    """
    Guarda un RunResult como JSON en output_dir/{experiment}_seed{seed}.json.

    Devuelve el Path del archivo creado.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{result.experiment}_seed{result.seed}.json"
    path = output_dir / filename

    # Serializar dataclass a dict
    data = dataclasses.asdict(result)
    # Añadir timestamp si no está presente
    if "timestamp" not in data or not data["timestamp"]:
        data["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return path
