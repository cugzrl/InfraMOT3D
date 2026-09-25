#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/third_party/3DMOTFormer"
COMMIT="9e88c243e4eacd75b11065d200e5ef3e616af474"
URL="https://codeload.github.com/dsx0511/3DMOTFormer/tar.gz/${COMMIT}"
if [ ! -f "${DEST}/model/model.py" ]; then
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
if ! grep -q '10 + int(num_classes)' "${DEST}/model/model.py"; then
  patch -p1 -d "${DEST}" < "${ROOT}/scripts/setup/patches/3dmotformer_v2x.patch"
fi
conda run -n track python -c 'import torch_geometric; from torch_geometric.nn import TransformerConv'
echo "3dmotformer_ok ${COMMIT}"
