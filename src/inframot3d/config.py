from pathlib import Path

import yaml


def _project_root(path):
    current = path if path.is_dir() else path.parent
    for candidate in [current, *current.parents]:
        if (candidate / "src" / "inframot3d").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise FileNotFoundError("找不到项目根目录 %s" % path)


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _project_root(path)
    config["_root"] = root
    for key in ("data_root", "converted_root", "output_root"):
        value = Path(config["project"][key])
        config["project"][key] = value if value.is_absolute() else root / value
    return config
