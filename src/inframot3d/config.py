from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    config["_root"] = root
    for key in ("data_root", "converted_root", "output_root"):
        value = Path(config["project"][key])
        config["project"][key] = value if value.is_absolute() else root / value
    return config
