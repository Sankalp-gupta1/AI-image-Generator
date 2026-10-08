# Recruiter Demo Guide

This file is written in simple language so the project can be explained clearly in an interview.

## 90-second demo flow

1. Start with the problem:
   "I wanted to build a practical Generative AI image application instead of only calling a model from a notebook."

2. Show the UI:
   - enter a short prompt
   - select SD-Turbo · Fast
   - keep one image and a square aspect ratio
   - generate the image

3. Show reproducibility:
   - open the metadata panel
   - point out model, seed, inference steps, device, image size and generation time
   - reuse the same seed to explain reproducibility

4. Show engineering controls:
   - change the style preset
   - explain negative prompting
   - mention that Stable Diffusion 1.5 exposes more meaningful guidance control

5. Close with implementation:
   "The application uses a Hugging Face Diffusers pipeline on top of PyTorch, lazy-loads the selected model, automatically detects CUDA, and serves the workflow through Gradio."

## Easy architecture explanation

```text
Prompt
  ↓
Validate + enrich prompt
  ↓
Select diffusion model
  ↓
Create seeded PyTorch generator
  ↓
Iterative denoising
  ↓
Decode latent representation
  ↓
Final image
```

## Questions you should be ready for

### What is a diffusion model?
A diffusion model learns to reverse a noise process. During generation it starts from random noise and repeatedly removes noise while being guided by the text prompt until an image is produced.

### What does the seed do?
The seed controls the initial random noise. Keeping the same seed and generation settings makes the output reproducible.

### What are inference steps?
They are the number of denoising iterations used to create the image. More steps can improve quality for some models but also increase generation time. Turbo models are trained to work with very few steps.

### What is guidance scale?
It controls how strongly the generation follows the text prompt. A higher value can make the model follow the prompt more aggressively, but very high values can also hurt image quality. SD-Turbo is designed to use guidance near zero.

### Why use negative prompts?
A negative prompt describes properties we want the model to avoid, such as blur, distortion or unwanted text.

### Why lazy-load models?
Diffusion models use a lot of memory. Keeping only the active model loaded reduces RAM/VRAM pressure and makes switching models safer.

### Why Gradio?
Gradio makes it fast to expose an ML inference function through a usable web interface, which is useful for prototypes, evaluation and demos.

### CPU vs GPU?
The same high-level pipeline can run on both, but GPU inference is much faster because diffusion repeatedly performs large tensor operations.

## Honest limitations

- the application uses pretrained diffusion models rather than training a foundation model from scratch
- first startup can be slow because model weights need to download
- CPU generation is slow
- generated images can still contain artifacts
- a production system would need stronger safety, observability, caching and scalable GPU serving
