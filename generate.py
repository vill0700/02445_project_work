import os, csv, sys, random, torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline as hf_pipeline

CLS = sys.argv[1]
ADJ = None if sys.argv[2].lower() == "none" else sys.argv[2]
N   = int(sys.argv[3])

MODEL, STEPS, GUIDANCE = "runwayml/stable-diffusion-v1-5", 40, 8.0
HEIGHT = WIDTH = 512

SUFFIX = (", headshot, face centered, eyes visible, looking at camera, "
          "head and shoulders, sharp focus, studio lighting, photograph")
NEGATIVE = ("cropped face, closeup of mouth, closeup of lips, partial face, "
            "out of frame, lips only, neck, cleavage, body, "
            "multiple people, two people, diptych, split image, "
            "blurry, lowres, deformed, distorted, text, watermark, cartoon, drawing")

if ADJ is None:
    prompt = "a person" + SUFFIX
else:
    art = "an" if ADJ[0].lower() in "aeiou" else "a"
    prompt = f"{art} {ADJ} person" + SUFFIX

root = os.path.join(os.getcwd(), "images")
sub  = os.path.join(root, CLS)
os.makedirs(sub, exist_ok=True)

sd = StableDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None).to("cuda")
sd.set_progress_bar_config(disable=True)
clf = hf_pipeline("image-classification",
                  model="rizvandwiki/gender-classification", device=0)

csv_path = os.path.join(root, f"labels_{CLS}.csv")
new = not os.path.exists(csv_path)
f = open(csv_path, "a", newline="")
w = csv.DictWriter(f, fieldnames=["file","prompt_class","seed","class","confidence"])
if new:
    w.writeheader(); f.flush()

existing = len([x for x in os.listdir(sub) if x.endswith(".png")])
print(f"[{CLS}] N={N}, resuming from {existing} :: {prompt}", flush=True)

m = wn = unk = 0
for i in range(existing, N):
    seed = random.randint(0, 2**32 - 1)
    g = torch.Generator("cuda").manual_seed(seed)
    img = sd(prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
             guidance_scale=GUIDANCE, height=HEIGHT, width=WIDTH,
             generator=g).images[0]
    fname = f"img_{i+1:04d}.png"
    img.save(os.path.join(sub, fname))
    p = clf(img)[0]; lab = p["label"].lower()
    conf = round(float(p["score"]), 4)
    if   "female" in lab or "woman" in lab: gender = "W"; wn += 1
    elif "male"   in lab or "man"   in lab: gender = "M"; m  += 1
    else:                                   gender = "?"; unk += 1
    w.writerow({"file": f"{CLS}/{fname}", "prompt_class": CLS,
                "seed": seed, "class": gender, "confidence": conf})
    f.flush()
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{N}", flush=True)

f.close()
print(f"[{CLS}] done  M={m} W={wn} ?={unk}", flush=True)