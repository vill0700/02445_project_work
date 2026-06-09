import csv
import json
from pathlib import Path
from typing import Annotated

import typer
from stable_diffusion_cpp import StableDiffusion

app = typer.Typer()

FLUX_DIR = Path("FLUX.1-schnell-GGUF")
GENDER_DIR = Path("gender-classification")

FLUX_MODELS = ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "f16"]
T5_MODELS = ["Q2_K", "Q3_K", "Q4_0", "Q4_K", "Q5_0", "Q5_1", "Q8_0", "f16"]


def _load_flux(flux_model: str, t5_model: str) -> StableDiffusion:
    return StableDiffusion(
        diffusion_model_path=str(FLUX_DIR / f"flux1-schnell-{flux_model}.gguf"),
        clip_l_path=str(FLUX_DIR / "clip_l.safetensors"),
        t5xxl_path=str(FLUX_DIR / f"t5xxl-{t5_model}.gguf"),
        vae_path=str(FLUX_DIR / "ae.safetensors"),
        keep_clip_on_cpu=True,
        keep_vae_on_cpu=True,
    )


def _next_batch_dir() -> Path:
    output = Path("output")
    output.mkdir(exist_ok=True)
    i = 1
    while (output / f"batch_{i}").exists():
        i += 1
    return output / f"batch_{i}"


@app.command()
def generate(
    prompts: Annotated[
        list[str],
        typer.Option("--prompt", "-p", help="Text prompt (repeat for multiple prompts)"),
    ],
    images_per_prompt: Annotated[
        int,
        typer.Option("--images-per-prompt", "-n", help="Number of images to generate per prompt"),
    ] = 1,
    seed: Annotated[
        int,
        typer.Option("--seed", "-s", help="Base random seed"),
    ] = 42,
    steps: Annotated[
        int,
        typer.Option("--steps", help="Number of sampling steps"),
    ] = 4,
    width: Annotated[
        int,
        typer.Option("--width", "-W", help="Image width in pixels"),
    ] = 1024,
    height: Annotated[
        int,
        typer.Option("--height", "-H", help="Image height in pixels"),
    ] = 1024,
    flux_model: Annotated[
        str,
        typer.Option("--flux-model", help=f"FLUX quantization variant: {', '.join(FLUX_MODELS)}"),
    ] = "Q4_0",
    t5_model: Annotated[
        str,
        typer.Option("--t5-model", help=f"T5 quantization variant: {', '.join(T5_MODELS)}"),
    ] = "Q2_K",
):
    """Generate images from text prompts using FLUX.1-schnell."""
    if flux_model not in FLUX_MODELS:
        typer.echo(f"Invalid --flux-model '{flux_model}'. Choose from: {', '.join(FLUX_MODELS)}", err=True)
        raise typer.Exit(1)
    if t5_model not in T5_MODELS:
        typer.echo(f"Invalid --t5-model '{t5_model}'. Choose from: {', '.join(T5_MODELS)}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading FLUX flux1-schnell-{flux_model}.gguf + T5 t5xxl-{t5_model}.gguf ...")
    model = _load_flux(flux_model, t5_model)
    batch_dir = _next_batch_dir()
    typer.echo(f"Saving to {batch_dir}/")

    records = []
    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = batch_dir / f"prompt_{prompt_idx}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(images_per_prompt):
            current_seed = seed + img_idx
            images = model.generate_image(
                prompt=prompt,
                seed=current_seed,
                width=width,
                height=height,
                sample_steps=steps,
                cfg_scale=1.0,
            )
            for image in images:
                path = prompt_dir / f"{current_seed}_{img_idx}.png"
                image.save(str(path))
                typer.echo(f"  {path}")
                records.append({
                    "path": str(path.relative_to(batch_dir)),
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "image_id": img_idx,
                    "seed": current_seed,
                    "flux_model": flux_model,
                    "t5_model": t5_model,
                })

    (batch_dir / "metadata.json").write_text(json.dumps({"images": records}, indent=2))


@app.command()
def classify(
    batch: Annotated[
        Path,
        typer.Argument(help="Path to batch directory, e.g. output/batch_1"),
    ],
):
    """Classify images in a batch by gender and save results to classifications.csv."""
    from PIL import Image
    from transformers import pipeline

    metadata_path = batch / "metadata.json"
    if not metadata_path.exists():
        typer.echo(f"No metadata.json found in {batch}", err=True)
        raise typer.Exit(1)

    records = json.loads(metadata_path.read_text())["images"]

    classifier = pipeline(
        "image-classification",
        model=str(GENDER_DIR),
        local_files_only=True,
    )

    csv_path = batch / "classifications.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "class", "prompt_id", "image_id", "seed"])
        for i, entry in enumerate(records):
            image = Image.open(batch / entry["path"])
            label = classifier(image)[0]["label"]  # "female" or "male"
            gender = "W" if label == "female" else "M"
            writer.writerow([i, gender, entry["prompt_id"], entry["image_id"], entry["seed"]])
            typer.echo(f"  [{i}] {Path(entry['path']).name} → {gender}")

    typer.echo(f"Saved {csv_path}")


if __name__ == "__main__":
    app()
