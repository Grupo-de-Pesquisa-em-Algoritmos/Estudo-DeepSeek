import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-R1",
    local_dir="/scratch/unioeste/hefesto/fabricio.tanquella/DeepSeek-R1",
    local_dir_use_symlinks=False,
    max_workers=8,
    token="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"
)
