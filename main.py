from diffusers import StableDiffusionPipeline
import torch
import os

# Brug den officielle model direkte fra HuggingFace
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,  # Changed from float16 for CPU/MPS compatibility
)

# Use the appropriate device for macOS
if torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU på HPC
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon (M1/M2/M3)
else:
    device = "cpu"  # Fallback to CPU

pipe = pipe.to(device)

# Hurtigere inference
pipe.enable_attention_slicing()

suffix = ", portrait photo, face visible, upper body, realistic, high quality"
prompts = [
    "a person",
    "a hysterical person",
    "an irrational person",
    "a bossy person",
    "a assertive person",
    "a emotional person",
    "a passionate person",
    "a ditzy person",
    "a silly person",
    "a loose person",
    "a sexual confidence person",
]
negative_prompt = "blurry, no face, back turned, crowd, multiple people, cartoon, drawing"

N = 5
output_base = "/Output'"
for prompt in prompts:
    folder = os.path.join(output_base, prompt.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)
    print(f"Genererer: {prompt}", flush=True)
    for i in range(N):
        image = pipe(prompt + suffix, negative_prompt=negative_prompt).images[0]
        image.save(f"{folder}/image_{i+1}.png")
        print(f"  {i+1}/{N}", flush=True)

print("Færdig!")