# 02445 Project Work

Statistical evaluation of AI learning models — DTU course 02445.

## Setup

**1. Install dependencies**

```bash
pip install uv
uv sync
```

**2. Download model weights**

Model weights are not stored in git. Fetch them from Hugging Face:

```bash
pip install huggingface_hub
python download_models.py
```

This downloads:
- [`second-state/FLUX.1-schnell-GGUF`](https://huggingface.co/second-state/FLUX.1-schnell-GGUF) → `FLUX.1-schnell-GGUF/`
- [`rizvandwiki/gender-classification`](https://huggingface.co/rizvandwiki/gender-classification) → `gender-classification/`

## Models

| Directory | Model | Purpose |
|---|---|---|
| `FLUX.1-schnell-GGUF/` | FLUX.1-schnell (GGUF quantized) | Text-to-image generation |
| `gender-classification/` | ViT-based gender classifier | Image classification |
