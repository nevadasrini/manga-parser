#!/usr/bin/env bash
# Fine-tune YOLO once per exported image variant (see export_all_image_variants.py).
# Run from manga-parser/ so relative paths resolve.

set -euo pipefail

EPOCHS="${EPOCHS:-100}"
VARIANTS="${VARIANTS:-raw preprocess_only preprocess_edges preprocess_repair_edges edges_repair_only}"

for v in ${VARIANTS}; do
  echo "=== training variant=${v} epochs=${EPOCHS} ==="
  python train_yolo.py \
    --data "data/processed/manga109_yolo_${v}/dataset.yaml" \
    --name "manga109_${v}" \
    --epochs "${EPOCHS}" \
    --exist-ok
done
