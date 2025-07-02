#!/bin/bash

set -e

dest_dir="../weights"
files=(
    "icon_detect/train_args.yaml"
    "icon_detect/model.pt"
    "icon_detect/model.yaml"
    "icon_caption/config.json"
    "icon_caption/generation_config.json"
    "icon_caption/model.safetensors"
)

echo "Start model download..."

mkdir -p "$dest_dir"

for file in "${files[@]}"; do
    huggingface-cli download microsoft/OmniParser-v2.0 "$file" --local-dir "$dest_dir"
done

echo "Renaming directory: icon_caption -> icon_caption_florence"

mv "$dest_dir/icon_caption" "$dest_dir/icon_caption_florence"

echo "Download completed."