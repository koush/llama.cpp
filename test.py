# pip install huggingface_hub hf_transfer
import os # Optional for faster downloading
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

name = "ubergarm/Qwen3-235B-A22B-GGUF"

from huggingface_hub import snapshot_download
snapshot_download(
  repo_id = name,
  local_dir = f"/mnt/scrypted-nvr/{name}",
  allow_patterns = ["*IQ3_K*"], # Select quant type UD-IQ1_S for 1.58bit
)

