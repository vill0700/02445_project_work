from pathlib import Path
from typing import Annotated

import typer
from stable_diffusion_cpp import StableDiffusion

app = typer.Typer()

FLUX_DIR = Path("FLUX.1-schnell-GGUF")


def _load_model() -> StableDiffusion:
    return StableDiffusion(
        diffusion_model_path=str(FLUX_DIR / "flux1-schnell-Q4_0.gguf"),
        clip_l_path=str(FLUX_DIR / "clip_l.safetensors"),
        t5xxl_path=str(FLUX_DIR / "t5xxl-Q4_0.gguf"),
        vae_path=str(FLUX_DIR / "ae.safetensors"),
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
):
    """Generate images from text prompts using FLUX.1-schnell."""
    model = _load_model()
    batch_dir = _next_batch_dir()
    typer.echo(f"Saving to {batch_dir}/")

    for prompt_idx, prompt in enumerate(prompts):
        prompt_dir = batch_dir / f"prompt_{prompt_idx}"
        prompt_dir.mkdir(parents=True, exist_ok=True)

        for img_idx in range(images_per_prompt):
            current_seed = seed + img_idx
            images = model.txt_to_img(
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


if __name__ == "__main__":
    app()
