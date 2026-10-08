from __future__ import annotations

STYLE_PRESETS = {
    "None": "",
    "Photorealistic": (
        "photorealistic professional photography, natural lighting, realistic textures, "
        "fine detail, balanced composition"
    ),
    "Cinematic": (
        "cinematic composition, dramatic natural lighting, film still, realistic depth, "
        "atmospheric detail"
    ),
    "Digital Art": (
        "high-detail digital art, imaginative environment, polished concept art, "
        "strong composition"
    ),
    "Anime": (
        "high-quality anime illustration, expressive composition, clean linework, "
        "detailed background"
    ),
    "Product Photography": (
        "premium product photography, studio lighting, commercial composition, "
        "clean background, crisp material detail"
    ),
}

QUALITY_BOOSTER = (
    "highly detailed, coherent composition, sharp subject, balanced lighting, "
    "professional visual quality"
)

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, deformed, duplicate, bad anatomy, extra limbs, "
    "poorly drawn hands, text, watermark, logo, oversaturated"
)


def validate_prompt(prompt: str) -> str:
    cleaned = (prompt or "").strip()
    if len(cleaned) < 3:
        raise ValueError("Please enter a meaningful prompt.")
    if len(cleaned) > 1200:
        raise ValueError("Prompt is too long. Keep it under 1200 characters for a clean demo.")
    return cleaned


def build_prompt(prompt: str, style: str, auto_enhance: bool = True) -> str:
    cleaned = validate_prompt(prompt)

    additions = []
    style_suffix = STYLE_PRESETS.get(style, "")
    if style_suffix:
        additions.append(style_suffix)
    if auto_enhance:
        additions.append(QUALITY_BOOSTER)

    if not additions:
        return cleaned

    return ", ".join([cleaned, *additions])
