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

## Usage

### Generate images

```bash
python main.py generate --prompt "a man in a suit" --prompt "a woman in a dress" --images-per-prompt 5 --seed 42
```

| Option | Short | Default | Description |
|---|---|---|---|
| `--prompt` | `-p` | *(required)* | Text prompt — repeat for multiple prompts |
| `--images-per-prompt` | `-n` | `1` | Number of images per prompt |
| `--seed` | `-s` | `42` | Base random seed (increments per image) |
| `--steps` | | `4` | Sampling steps |
| `--width` | `-W` | `1024` | Image width in pixels |
| `--height` | `-H` | `1024` | Image height in pixels |

Output is saved to `output/batch_N/prompt_M/{seed}_{id}.png`. The batch number increments automatically on each run. A `metadata.json` is written to the batch directory recording the prompt, seed, and IDs for every image.

### Classify images

```bash
python main.py classify output/batch_1
```

Reads `metadata.json` from the batch directory, runs the gender classifier on each image, and saves `classifications.csv` with columns:

| Column | Description |
|---|---|
| `i` | Row index |
| `class` | Classifier result — `M` (male) or `W` (female) |
| `prompt_id` | Prompt index (0-based) |
| `image_id` | Image index within the prompt |
| `seed` | Random seed used to generate the image |

## Models

| Directory | Model | Purpose |
|---|---|---|
| `FLUX.1-schnell-GGUF/` | FLUX.1-schnell (GGUF quantized) | Text-to-image generation |
| `gender-classification/` | ViT-based gender classifier | Image classification |
