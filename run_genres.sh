#!/bin/bash

# 特点：
# 自动轮询 GPU
# 所有任务 并行
# GPU 间互不影响
# genre 数量 > GPU 数量也没问题

# ===== 需要跑的 genre =====
genres=(
  blues
  classical  
  country  
  electronic  
  jazz  
  rap
  rnb
  rock  
  world
)

# ===== 可用 GPU =====
gpus=(
  0
  1
  2
  4
  5
  6
  7
)

num_gpus=${#gpus[@]}
idx=0

for genre in "${genres[@]}"; do
  gpu_id=${gpus[$((idx % num_gpus))]}
  echo "[DISPATCH] genre=${genre} -> GPU=${gpu_id}"

  bash infer_gtsinger_one.sh "${genre}" "${gpu_id}" &

  idx=$((idx + 1))
done

wait
echo "[ALL DONE]"
