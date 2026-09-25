#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/third_party/FastPoly"
COMMIT="694dec303db0831cbed4f8cac3f39dc1afe397ba"
URL="https://codeload.github.com/lixiaoyu2000/FastPoly/tar.gz/${COMMIT}"
if [ ! -f "${DEST}/tracking/nusc_tracker.py" ]; then
  TMP="$(mktemp -d)"
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    curl -L --fail --retry 3 -o "${TMP}/src.tar.gz" "${URL}"
  tar -xzf "${TMP}/src.tar.gz" -C "${TMP}"
  INNER="$(find "${TMP}" -mindepth 1 -maxdepth 1 -type d | head -1)"
  mkdir -p "${ROOT}/third_party"
  rm -rf "${DEST}"
  mv "${INNER}" "${DEST}"
  rm -rf "${TMP}"
fi
printf '%s\n' "${COMMIT}" > "${DEST}/COMMIT"
python3 - "${DEST}/motion_module/motion_model.py" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
old = "assert q.radians == det_box.yaw"
new = "assert abs(float(q.radians) - float(det_box.yaw)) < 1e-5"
if old not in text and new not in text:
    raise SystemExit("Fast-Poly四元数检查与预期不一致")
path.write_text(text.replace(old, new), encoding="utf-8")
PY
conda run -n track python -c 'import numba, sympy, pyquaternion, lap, nuscenes'
echo "fastpoly_ok ${COMMIT}"
