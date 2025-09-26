# Segmentation + ASR
# audio_dir="/data7/fwh/syn_10k_0907"
audio_dir="wavs"
metadata_file="${audio_dir}/metadata.json"

# python scripts/process_ch.py -i "$audio_dir" -o "$metadata_file"

# Annotation
# python inference/stars.py \
#     --ckpt checkpoints/stars_chinese/model_ckpt_steps_200000.ckpt \
#     --config configs/stars_chinese.yaml \
#     --phset chinese_phone_set.json \
#     --metadata "$metadata_file" \
#     -o "$audio_dir"

# get music score
json_path="${audio_dir}/output.json"
# echo "$json_path"
python json2score.py \
    --json_path "$json_path"