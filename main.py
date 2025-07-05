import torch
from diffusers import StableDiffusionPipeline
import gradio as gr

# Model ID from Hugging Face
model_id = "CompVis/stable-diffusion-v1-4"

# Load the pipeline for CPU (no GPU)
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32   # For CPU, use float32
)

# Move model to CPU
pipe = pipe.to("cpu")

# Function to generate image from text prompt
def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Launch Gradio UI
gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Jaise likho: A cyberpunk cat playing guitar..."),
    outputs="image",
    title="🎨 AI Image Generator (CPU Version)",
    description="Stable Diffusion ke zariye text se image banayein – bina GPU ke bhi but this is time consuming 2-3min! 🐌",
    theme="default"
).launch()
