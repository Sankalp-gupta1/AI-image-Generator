from __future__ import annotations

import gc
import os
import secrets
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image

from prompt_utils import (
    DEFAULT_NEGATIVE_PROMPT,
    STYLE_PRESETS,
    build_prompt,
    validate_prompt,
)


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    label: str
    default_steps: int
    max_steps: int
    default_guidance: float
    force_guidance: float | None = None


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "SD-Turbo · Fast": ModelConfig(
        model_id="stabilityai/sd-turbo",
        label="SD-Turbo",
        default_steps=2,
        max_steps=4,
        default_guidance=0.0,
        force_guidance=0.0,
    ),
    "Stable Diffusion 1.5 · Quality": ModelConfig(
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        label="Stable Diffusion 1.5",
        default_steps=25,
        max_steps=40,
        default_guidance=7.5,
    ),
}

ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "Square · 1:1": (512, 512),
    "Landscape · 3:2": (768, 512),
    "Portrait · 2:3": (512, 768),
}


class GeneratorEngine:
    """Lazy-loads diffusion pipelines so the demo stays memory friendly."""

    def __init__(self) -> None:
        self.pipeline = None
        self.loaded_model_key: str | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def unload(self) -> None:
        self.pipeline = None
        self.loaded_model_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load(self, model_key: str) -> ModelConfig:
        if model_key not in MODEL_CONFIGS:
            raise ValueError("Unknown model selection.")

        if self.pipeline is not None and self.loaded_model_key == model_key:
            return MODEL_CONFIGS[model_key]

        self.unload()
        config = MODEL_CONFIGS[model_key]

        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }

        cache_dir = os.getenv("HF_HOME")
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            config.model_id,
            **load_kwargs,
        )
        self.pipeline = self.pipeline.to(self.device)

        if hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
        if hasattr(self.pipeline, "set_progress_bar_config"):
            self.pipeline.set_progress_bar_config(disable=True)

        self.loaded_model_key = model_key
        return config

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        style: str,
        model_key: str,
        aspect_ratio: str,
        steps: int,
        guidance: float,
        seed: int,
        image_count: int,
        auto_enhance: bool,
    ):
        validate_prompt(prompt)

        if aspect_ratio not in ASPECT_RATIOS:
            raise ValueError("Unknown aspect ratio.")

        config = self.load(model_key)
        width, height = ASPECT_RATIOS[aspect_ratio]

        final_prompt = build_prompt(
            prompt=prompt,
            style=style,
            auto_enhance=auto_enhance,
        )

        actual_steps = max(1, min(int(steps), config.max_steps))
        actual_guidance = (
            config.force_guidance
            if config.force_guidance is not None
            else float(guidance)
        )

        resolved_seed = int(seed)
        if resolved_seed < 0:
            resolved_seed = secrets.randbelow(2_147_483_647)

        generators = [
            torch.Generator(device=self.device).manual_seed(resolved_seed + index)
            for index in range(int(image_count))
        ]

        negative = (negative_prompt or "").strip() or None
        if actual_guidance == 0:
            negative = None

        started = time.perf_counter()
        with torch.inference_mode():
            result = self.pipeline(
                prompt=final_prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_inference_steps=actual_steps,
                guidance_scale=actual_guidance,
                num_images_per_prompt=int(image_count),
                generator=generators,
            )

        elapsed = time.perf_counter() - started
        images = result.images

        metadata = {
            "model": config.label,
            "model_id": config.model_id,
            "device": self.device.upper(),
            "seed": resolved_seed,
            "image_count": len(images),
            "size": f"{width}x{height}",
            "steps": actual_steps,
            "guidance_scale": actual_guidance,
            "style": style,
            "prompt_used": final_prompt,
            "negative_prompt_used": negative or "Not used by this model/configuration",
            "generation_time_seconds": round(elapsed, 2),
        }

        status = (
            f"✅ Generated **{len(images)} image(s)** in **{elapsed:.1f}s** "
            f"using **{config.label}** on **{self.device.upper()}**. "
            f"Seed: **{resolved_seed}**"
        )

        if self.device == "cpu":
            status += "  \n⚠️ CPU mode detected. For a live demo, a Colab GPU is strongly recommended."

        return images, status, metadata


ENGINE = GeneratorEngine()


def generate_ui(
    prompt,
    negative_prompt,
    style,
    model_key,
    aspect_ratio,
    steps,
    guidance,
    seed,
    image_count,
    auto_enhance,
):
    try:
        return ENGINE.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            style=style,
            model_key=model_key,
            aspect_ratio=aspect_ratio,
            steps=steps,
            guidance=guidance,
            seed=seed,
            image_count=image_count,
            auto_enhance=auto_enhance,
        )
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            ENGINE.unload()
            raise gr.Error(
                "GPU memory ran out. Try one image, a smaller aspect ratio, or the fast model."
            ) from exc
        raise gr.Error(f"Generation failed: {message}") from exc
    except Exception as exc:
        raise gr.Error(f"Generation failed: {exc}") from exc


def build_demo() -> gr.Blocks:
    css = """
    .hero {
        text-align: center;
        padding: 8px 0 14px;
    }
    .hero h1 {
        font-size: 2.35rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        opacity: 0.78;
        font-size: 1.02rem;
    }
    .feature-row {
        text-align: center;
        opacity: 0.84;
        margin-bottom: 12px;
    }
    """

    with gr.Blocks(
        title="GenVision Studio",
        theme=gr.themes.Soft(),
        css=css,
    ) as demo:
        gr.Markdown(
            """
            <div class="hero">
              <h1>✨ GenVision Studio</h1>
              <p>Prompt → Diffusion Model → Reproducible AI Image</p>
            </div>
            <div class="feature-row">
              Fast + quality model modes · Style presets · Negative prompts · Seed control · GPU/CPU auto-detection
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Describe the image",
                    placeholder="Example: a futuristic electric bike in a rainy neon city, cinematic photography",
                    lines=5,
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEGATIVE_PROMPT,
                    lines=3,
                )
                with gr.Row():
                    style = gr.Dropdown(
                        choices=list(STYLE_PRESETS.keys()),
                        value="Photorealistic",
                        label="Style preset",
                    )
                    auto_enhance = gr.Checkbox(
                        value=True,
                        label="Smart prompt enhancement",
                    )

            with gr.Column(scale=4):
                model_key = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="SD-Turbo · Fast",
                    label="Generation model",
                )
                aspect_ratio = gr.Radio(
                    choices=list(ASPECT_RATIOS.keys()),
                    value="Square · 1:1",
                    label="Aspect ratio",
                )
                with gr.Row():
                    steps = gr.Slider(
                        minimum=1,
                        maximum=40,
                        value=4,
                        step=1,
                        label="Inference steps",
                    )
                    guidance = gr.Slider(
                        minimum=0,
                        maximum=15,
                        value=7.5,
                        step=0.5,
                        label="Guidance scale",
                    )
                with gr.Row():
                    seed = gr.Number(
                        value=-1,
                        precision=0,
                        label="Seed (-1 = random)",
                    )
                    image_count = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                        label="Number of images",
                    )

        generate_button = gr.Button(
            "Generate Image",
            variant="primary",
            size="lg",
        )

        status = gr.Markdown(
            "Ready. The first generation can take longer because the model has to load."
        )
        gallery = gr.Gallery(
            label="Generated images",
            columns=2,
            rows=2,
            object_fit="contain",
            height=620,
        )

        with gr.Accordion("Generation metadata / reproducibility", open=False):
            metadata = gr.JSON(label="Run details")

        gr.Markdown("### Quick demo prompts")
        gr.Examples(
            examples=[
                [
                    "A premium electric sports car parked under rain-soaked neon streets in Tokyo, dramatic reflections",
                    "Cinematic",
                ],
                [
                    "A realistic Indian space scientist inside a modern mission control room, documentary photography",
                    "Photorealistic",
                ],
                [
                    "A clean product advertisement for futuristic wireless headphones on a minimal studio pedestal",
                    "Product Photography",
                ],
                [
                    "A floating eco-friendly city above the clouds with gardens, bridges and solar architecture",
                    "Digital Art",
                ],
            ],
            inputs=[prompt, style],
        )

        inputs = [
            prompt,
            negative_prompt,
            style,
            model_key,
            aspect_ratio,
            steps,
            guidance,
            seed,
            image_count,
            auto_enhance,
        ]
        outputs = [gallery, status, metadata]

        generate_button.click(
            fn=generate_ui,
            inputs=inputs,
            outputs=outputs,
            show_progress="full",
        )
        prompt.submit(
            fn=generate_ui,
            inputs=inputs,
            outputs=outputs,
            show_progress="full",
        )

        gr.Markdown(
            """
            **Demo note:** SD-Turbo is optimized for very few steps and internally forces guidance to 0.
            Choose Stable Diffusion 1.5 · Quality when you want more control over steps and guidance.
            """
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    server_port = int(os.getenv("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share,
    )
