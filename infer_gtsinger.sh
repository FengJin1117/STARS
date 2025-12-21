
export CUDA_VISIBLE_DEVICES=3

# 1. Segmentation + ASR
genre="pop" # 请输入genre
audio_dir="../benchdata/suno_vocal/${genre}"
output_dir="../benchdata/suno_score_fix/${genre}"

# opencpop处理
# audio_dir="../data/opencpop_test_stars/testset"
# output_dir="../data/opencpop_test_stars"

metadata_file="${output_dir}/metadata.json"
export PYTHONPATH=.

python scripts/process_ch.py -i "$audio_dir" -o "$metadata_file"
python filter.py --metadata "$metadata_file" 

# 2. Annotation
python inference/stars.py \
    --ckpt checkpoints/stars_chinese/model_ckpt_steps_200000.ckpt \
    --config configs/stars_chinese.yaml \
    --phset chinese_phone_set.json \
    --metadata "$metadata_file" \
    -o "$output_dir" \
    --no_save_textgrid --no_save_midi

# 3. get music score
python gtsinger.py "${output_dir}/output.json"
python phone_mapping.py --score_file "${output_dir}/gtsinger.txt"