"""Download model weights from Hugging Face. Run once before using the project."""

from huggingface_hub import snapshot_download

FLUX_REPO = "second-state/FLUX.1-schnell-GGUF"
GENDER_REPO = "rizvandwiki/gender-classification"

print(f"Downloading FLUX.1-schnell GGUF models from {FLUX_REPO} ...")
snapshot_download(
    repo_id=FLUX_REPO,
    local_dir="FLUX.1-schnell-GGUF",
    ignore_patterns=["*.md", ".gitattributes"],
)

print(f"Downloading gender-classification model from {GENDER_REPO} ...")
snapshot_download(
    repo_id=GENDER_REPO,
    local_dir="gender-classification",
    ignore_patterns=["*.md", ".gitattributes", "images/"],
)

print("Done.")
