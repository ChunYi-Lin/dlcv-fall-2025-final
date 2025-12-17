from huggingface_hub import snapshot_download
from argparse import ArgumentParser
import os

def parse_args():
    parser = ArgumentParser(description="Download a Hugging Face model snapshot locally.")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model repo id on the Hugging Face Hub",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="./Qwen2.5-7B-Instruct",
        help="Directory to save the model snapshot",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision (branch/tag/commit)",
    )
    parser.add_argument(
        "--ignore_patterns",
        type=str,
        nargs="*",
        default=["*.msgpack", "*.h5", "*.ot"],
        help="Glob patterns to skip",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    local_model_dir = os.path.abspath(args.local_dir)
    os.makedirs(local_model_dir, exist_ok=True)

    print(f"Downloading {args.repo_id} to {local_model_dir}...")

    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=local_model_dir,
        ignore_patterns=args.ignore_patterns,
    )

    print("Download complete.")
