---
base_model: black-forest-labs/FLUX.1-schnell
license: apache-2.0
model_creator: black-forest-labs
model_name: FLUX.1-schnell
quantized_by: Second State Inc.
language:
- en
tags:
- text-to-image
- image-generation
- flux
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://github.com/LlamaEdge/LlamaEdge/raw/dev/assets/logo.svg" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# FLUX.1-schnell-GGUF

## Original Model

[black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

## Run with `sd-api-server`

- sd-api-server version: [0.1.4](https://github.com/LlamaEdge/sd-api-server/releases/tag/0.1.4)

- Run as LlamaEdge service

  ```bash
  wasmedge --dir .:. sd-api-server.wasm \
    --model-name flux1-schnell \
    --diffusion-model flux1-schnell-Q4_0.gguf \
    --vae ae.safetensors \
    --clip-l clip_l.safetensors \
    --t5xxl t5xxl-Q8_0.gguf
  ```

  *For details, see https://github.com/LlamaEdge/sd-api-server/blob/main/examples/flux.md*

## Quantized GGUF Models

| Name | Quant method | Bits | Size | Use case |
| ---- | ---- | ---- | ---- | ----- |
| [ae-f16.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/ae-f16.gguf)   | f16 | 16 | 168 MB | |
| [ae.safetensors](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/ae.safetensors)   | f32 | 32 | 335 MB | |
| [clip_l-Q8_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/clip_l-Q8_0.gguf) | Q8_0 | 8 | 131 MB | |
| [clip_l.safetensors](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/clip_l.safetensors)   | f16 | 16 | 246 MB | |
| [flux1-schnell-Q4_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell-Q4_0.gguf) | Q4_0 | 4 | 6.69 GB | |
| [flux1-schnell-Q4_1.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell-Q4_1.gguf) | Q4_1 | 4 | 7.43 GB | |
| [flux1-schnell-Q5_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell-Q5_0.gguf) | Q5_0 | 5 | 8.18 GB | |
| [flux1-schnell-Q5_1.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell-Q5_1.gguf) | Q5_1 | 5 | 8.92 GB | |
| [flux1-schnell-Q8_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell-Q8_0.gguf) | Q8_0 | 8 | 12.6 GB | |
| [flux1-schnell.safetensors](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/flux1-schnell.safetensors)  | f16 | 16 | 23.8 GB | |
| [t5xxl-Q2_K.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q2_K.gguf) | Q2_K | 2 | 1.61 GB | |
| [t5xxl-Q3_K.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q3_K.gguf) | Q3_K | 3 | 2.10 GB | |
| [t5xxl-Q4_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q4_0.gguf) | Q4_0 | 4 | 2.75 GB | |
| [t5xxl-Q4_K.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q4_K.gguf) | Q4_K | 4 | 2.75 GB | |
| [t5xxl-Q5_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q5_0.gguf) | Q5_0 | 5 | 3.36 GB | |
| [t5xxl-Q5_1.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q5_1.gguf) | Q5_1 | 5 | 3.67 GB | |
| [t5xxl-Q8_0.gguf](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl-Q8_0.gguf) | Q8_0 | 8 | 5.20 GB | |
| [t5xxl_fp16.safetensors](https://huggingface.co/second-state/FLUX.1-schnell-GGUF/blob/main/t5xxl_fp16.safetensors)   | f16 | 16 | 9.79 GB | |

**Quantized with stable-diffusion.cpp `master-64d231f`.**