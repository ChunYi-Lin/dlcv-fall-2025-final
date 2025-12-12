from huggingface_hub import snapshot_download
import os

# Define where you want to save the model
local_model_dir = "./Qwen2.5-7B-Instruct"

print(f"Downloading Qwen2.5-7B-Instruct to {local_model_dir}...")

# This will download all necessary files (weights, tokenizer configs, etc.)
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir=local_model_dir,
    ignore_patterns=["*.msgpack", "*.h5", "*.ot"] # Optional: skip non-safetensors formats
)

print("Download complete.")