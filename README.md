# 02445 Project Work

Statistical evaluation of AI learning models — DTU course 02445.

## Setup

**1. Install system dependencies**

```bash
sudo apt install build-essential cmake nvidia-cuda-toolkit
```

**2. Install Python dependencies (with GPU support)**

```bash
pip install uv
SKBUILD_CMAKE_ARGS="-DSD_CUDA=ON" uv sync
```

> **CPU-only fallback:** skip the `SKBUILD_CMAKE_ARGS` prefix if you don't have an NVIDIA GPU:
> ```bash
> uv sync
> ```

**3. Download model weights**

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
| `--flux-model` | | `Q4_0` | FLUX quantization: `Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `f16` |
| `--t5-model` | | `Q2_K` | T5 quantization: `Q2_K` `Q3_K` `Q4_0` `Q4_K` `Q5_0` `Q5_1` `Q8_0` `f16` |

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
