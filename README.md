# GenVision Studio — Generative AI Image Studio

> **Live Demo:** https://sankalp-gupta1.github.io/AI-image-Generator/  
> **GitHub:** https://github.com/Sankalp-gupta1/AI-image-Generator  
> **Status:** CI passing · GitHub Pages deployment passing

A practical **text-to-image Generative AI application** built with **Python, PyTorch, Hugging Face Diffusers, Stable Diffusion and Gradio**.

The project is designed as a recruiter-ready demo: a user enters a natural-language prompt, chooses generation controls, and receives reproducible AI-generated images through a clean web interface.

## Why this project matters

Most AI image demos stop at a single prompt box. GenVision Studio adds the engineering controls needed to explain how an image generation system behaves in practice:

- two generation modes: **SD-Turbo** for speed and **Stable Diffusion 1.5** for more control
- automatic **GPU / CPU detection**
- reusable **style presets**
- **negative prompting**
- deterministic **seed control**
- adjustable **inference steps** and **guidance scale**
- multiple output images per run
- portrait, landscape and square generation
- generation time + model metadata for reproducibility
- lazy model loading and memory cleanup
- a polished **Gradio** interface
- lightweight unit tests + GitHub Actions CI

No paid image-generation API is required.

---

## Demo flow

```text
User Prompt
    │
    ▼
Prompt Validation
    │
    ├── Style Preset
    ├── Prompt Enhancement
    └── Negative Prompt
    │
    ▼
Diffusion Pipeline
    │
    ├── Model Selection
    ├── Seed
    ├── Inference Steps
    ├── Guidance Scale
    └── Aspect Ratio
    │
    ▼
PyTorch Inference
    │
    ▼
Generated Image(s)
    │
    ├── Gradio Gallery
    └── Reproducibility Metadata
```

## Tech stack

| Technology | Purpose |
| --- | --- |
| Python | Application logic |
| PyTorch | Tensor operations and model inference |
| Hugging Face Diffusers | Diffusion model pipelines |
| Stable Diffusion / SD-Turbo | Image generation |
| Gradio | Interactive web UI |
| Safetensors | Efficient model weight loading |
| Pytest | Unit testing |
| GitHub Actions | Continuous integration |

## Run locally

> A CUDA-capable GPU is recommended. CPU mode works, but image generation can be slow.

```bash
git clone https://github.com/Sankalp-gupta1/AI-image-Generator.git
cd AI-image-Generator

python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### macOS / Linux

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open:

```text
http://127.0.0.1:7860
```

## Google Colab demo

For an interview/demo, a Colab GPU is the easiest option.

1. Open a new Colab notebook.
2. Set **Runtime → Change runtime type → GPU**.
3. Clone this repository.
4. Install the requirements.
5. Run with Gradio sharing enabled.

```bash
!git clone https://github.com/Sankalp-gupta1/AI-image-Generator.git
%cd AI-image-Generator
!pip install -r requirements.txt
%env GRADIO_SHARE=true
!python main.py
```

The terminal will print a temporary public Gradio link that can be used for the live demo.

## Model modes

### SD-Turbo · Fast

Best for a quick recruiter demo.

- optimized for only a few inference steps
- guidance is forced to `0`
- faster feedback loop

### Stable Diffusion 1.5 · Quality

Best when explaining diffusion controls.

- more inference steps
- configurable classifier-free guidance
- useful for showing the effect of seed, prompt and generation parameters

## Project structure

```text
AI-image-Generator/
├── app.py
├── main.py
├── prompt_utils.py
├── requirements.txt
├── README.md
├── DEMO_GUIDE.md
├── tests/
│   └── test_prompt_utils.py
└── .github/
    └── workflows/
        └── tests.yml
```

## How reproducibility works

The same prompt + model + seed + generation settings should produce the same or very similar result on the same software stack.

The UI exposes the seed and returns generation metadata so a result can be explained and recreated instead of being treated like a black box.

## Practical engineering choices

**Lazy model loading:** only the selected model is kept in memory.

**GPU/CPU fallback:** the application detects CUDA automatically and chooses an appropriate tensor dtype.

**Memory handling:** attention/decoder slicing is enabled where supported, and GPU cache cleanup is performed when switching models.

**Input validation:** empty or extremely long prompts are rejected before inference.

**Graceful failure:** out-of-memory errors produce an understandable message instead of crashing silently.

## Tests

The prompt layer is tested independently from the large diffusion models:

```bash
pytest -q
```

This keeps CI fast while still verifying prompt validation and deterministic prompt-building behavior.

## What I would improve next

- image-to-image / controlled editing
- ControlNet for pose, depth or edge guidance
- LoRA adapter support
- persistent generation history
- moderation / policy layer for production use
- deployment on a GPU-backed Hugging Face Space or cloud inference service

## Interview explanation

A concise walkthrough and common viva questions are available in [DEMO_GUIDE.md](DEMO_GUIDE.md).

## Notes

Model weights are downloaded from Hugging Face on first use, so the first run takes longer. Runtime and memory usage depend on the selected model, hardware and image size.
