import os, torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

prompt = "a person, portrait photo, face visible, upper body, realistic, high quality"
g = torch.Generator("cuda").manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=g).images[0]

out_dir = os.path.join(os.getcwd(), "dataset", "test")
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, "image_001.png")
image.save(path)
print("saved", path, flush=True)