import subprocess
import csv
from pathlib import Path

# Folder where this Python file lives
BASE_DIR = Path(__file__).resolve().parent

# -----------------------------
# SETTINGS
# -----------------------------

SD_CLI = BASE_DIR / "build" / "bin" / "sd-cli"

# IMPORTANT:
# This is the REAL 1.5 GB model file.
MODEL_PATH = BASE_DIR.parent / "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf"

OUTPUT_ROOT = BASE_DIR / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

IMAGES_PER_PROMPT = 10

PROMPTS = [
    "a realistic close-up portrait of a firefighter wearing protective gear, sharp focus, detailed face, high quality, professional photography",
]

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, bad hands, extra fingers, "
    "deformed, ugly, watermark, text, cropped"
)

STEPS = 40
CFG_SCALE = 7.5
WIDTH = 512
HEIGHT = 512

# -----------------------------
# FIND NEXT BATCH FOLDER
# -----------------------------

def get_next_batch_dir(output_root: Path) -> Path:
    existing_batch_numbers = []

    for item in output_root.iterdir():
        if item.is_dir() and item.name.startswith("batch_"):
            try:
                number = int(item.name.replace("batch_", ""))
                existing_batch_numbers.append(number)
            except ValueError:
                pass

    next_number = max(existing_batch_numbers, default=0) + 1
    return output_root / f"batch_{next_number:03d}"


BATCH_DIR = get_next_batch_dir(OUTPUT_ROOT)
BATCH_DIR.mkdir(exist_ok=True)

# -----------------------------
# CHECKS
# -----------------------------

if not SD_CLI.exists():
    raise FileNotFoundError(f"Could not find sd-cli at: {SD_CLI}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Could not find model at: {MODEL_PATH}")

model_size_gb = MODEL_PATH.stat().st_size / (1024 ** 3)

print(f"Using sd-cli: {SD_CLI}")
print(f"Using model: {MODEL_PATH}")
print(f"Model size: {model_size_gb:.2f} GB")
print(f"Saving this run in: {BATCH_DIR}")

if model_size_gb < 1:
    raise RuntimeError(
        "Model file is too small. You are probably using the wrong Git LFS pointer file."
    )

# -----------------------------
# GENERATE IMAGES
# -----------------------------

metadata_rows = []

for prompt_index, prompt in enumerate(PROMPTS, start=1):
    prompt_dir = BATCH_DIR / f"prompt_{prompt_index:02d}"
    prompt_dir.mkdir(exist_ok=True)

    for image_number in range(1, IMAGES_PER_PROMPT + 1):
        seed = image_number
        output_file = prompt_dir / f"image_{image_number:02d}.png"

        command = [
            str(SD_CLI),
            "-m", str(MODEL_PATH),
            "-p", prompt,
            "-n", NEGATIVE_PROMPT,
            "--steps", str(STEPS),
            "--cfg-scale", str(CFG_SCALE),
            "-W", str(WIDTH),
            "-H", str(HEIGHT),
            "-o", str(output_file),
            "--seed", str(seed),
        ]

        print(f"\nGenerating image {image_number}/{IMAGES_PER_PROMPT}")
        print(f"Prompt: {prompt}")
        print(f"Output: {output_file}")

        subprocess.run(command, check=True)

        metadata_rows.append({
            "batch": BATCH_DIR.name,
            "prompt_index": prompt_index,
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "image_number": image_number,
            "seed": seed,
            "steps": STEPS,
            "cfg_scale": CFG_SCALE,
            "width": WIDTH,
            "height": HEIGHT,
            "file_path": str(output_file),
            "model": str(MODEL_PATH),
        })

# -----------------------------
# SAVE METADATA
# -----------------------------

metadata_file = BATCH_DIR / "metadata.csv"

with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "batch",
            "prompt_index",
            "prompt",
            "negative_prompt",
            "image_number",
            "seed",
            "steps",
            "cfg_scale",
            "width",
            "height",
            "file_path",
            "model",
        ],
    )
    writer.writeheader()
    writer.writerows(metadata_rows)

print("\nDone.")
print(f"Images saved in: {BATCH_DIR}")
print(f"Metadata saved in: {metadata_file}")