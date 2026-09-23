#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party/GRAE-3DMOT"
COMMIT="63def8bde5e199a4e77fdf4fab76a4b3511fe132"
if [ ! -d "${DEST}/.git" ]; then
  git clone https://github.com/altkddhfcjs/GRAE-3DMOT.git "${DEST}"
fi
git -C "${DEST}" fetch --quiet origin "${COMMIT}" || true
git -C "${DEST}" checkout --quiet "${COMMIT}"
python3 - "${DEST}/models/main.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = """    def __init__(self,
                 in_channels=256,
                 layers=3,
                 device=\"cuda:0\"
                 ):
        super().__init__()
        self.d_model = in_channels
        self.delta = 5  # track age
        self.layers = layers

        self.point_range = 51.2
"""
new = """    def __init__(self,
                 in_channels=256,
                 layers=3,
                 device=\"cuda:0\",
                 num_classes=7
                 ):
        super().__init__()
        self.d_model = in_channels
        self.delta = 5  # track age
        self.layers = layers
        # 类别数决定几何特征维度，默认7与官方nuScenes一致
        self.num_classes = int(num_classes)
        self.coord_dim = 11 + self.num_classes
        self.spatial_dim = 12 + self.num_classes

        self.point_range = 51.2
"""
if "self.coord_dim" not in text:
    if old not in text:
        raise SystemExit("GRAE初始化代码与预期不一致")
    text = text.replace(old, new, 1)
    text = text.replace(
        "self.coord_proj = MLP(18, self.d_model // 4, self.d_model)\n        self.spatial_proj = MLP(19, self.d_model // 4, self.d_model)",
        "self.coord_proj = MLP(self.coord_dim, self.d_model // 4, self.d_model)\n        self.spatial_proj = MLP(self.spatial_dim, self.d_model // 4, self.d_model)",
        1,
    )
    path.write_text(text, encoding="utf-8")
PY
conda run -n track python -c 'import lap, fvcore'
echo "grae_ok ${COMMIT}"
