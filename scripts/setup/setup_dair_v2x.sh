#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/third_party/DAIR-V2X"
COMMIT="c885c54af0c34bc515fa9ca8b5e8fda76a15462c"
if [[ -f "${DEST}/v2x/AB3DMOT_plugin/scripts/KITTI/evaluate.py" && -f "${DEST}/COMMIT" ]]; then
  echo "已存在 ${DEST} $(cat "${DEST}/COMMIT")"
  exit 0
fi
rm -rf "${DEST}"
git clone --filter=blob:none --sparse --depth 1 https://github.com/AIR-THU/DAIR-V2X.git "${DEST}"
git -C "${DEST}" sparse-checkout set v2x tools/dataset_converter
git -C "${DEST}" fetch --depth 1 origin "${COMMIT}"
git -C "${DEST}" checkout "${COMMIT}"
printf '%s\n' "${COMMIT}" > "${DEST}/COMMIT"
echo "已固定 ${COMMIT}"
