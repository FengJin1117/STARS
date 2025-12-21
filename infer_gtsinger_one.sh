#!/bin/bash
set -e

# ===== 参数 =====
GENRE=$1
GPU_ID=$2

if [ -z "$GENRE" ] || [ -z "$GPU_ID" ]; then
  echo "Usage: bash infer_gtsinger_one.sh <genre> <gpu_id>"
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONPATH=.

echo "[INFO] Start genre=${GENRE} on GPU=${GPU_ID}"

audio_dir="../benchdata/suno_vocal/${GENRE}"
output_dir="../benchdata/suno_score_fix/${GENRE}"
metadata_file="${output_dir}/metadata.json"

mkdir -p "${output_dir}"

# ===== 1. Segmentation + ASR =====
python scripts/process_ch.py -i "$audio_dir" -o "$metadata_file"
python filter.py --metadata "$metadata_file"

# ===== 2. Annotation =====
python inference/stars.py \
  --ckpt checkpoints/stars_chinese/model_ckpt_steps_200000.ckpt \
  --config configs/stars_chinese.yaml \
  --phset chinese_phone_set.json \
  --metadata "$metadata_file" \
  -o "$output_dir" \
  --no_save_textgrid \
  --no_save_midi

# ===== 3. Get music score =====
python gtsinger.py "${output_dir}/output.json"
python phone_mapping.py --score_file "${output_dir}/gtsinger.txt"

echo "[INFO] Done genre=${GENRE} on GPU=${GPU_ID}"
