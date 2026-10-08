import pytest

from prompt_utils import build_prompt, validate_prompt


def test_validate_prompt_rejects_empty_text():
    with pytest.raises(ValueError):
        validate_prompt("  ")


def test_validate_prompt_returns_cleaned_text():
    assert validate_prompt("  a red bicycle  ") == "a red bicycle"


def test_build_prompt_adds_style_and_quality_booster():
    result = build_prompt(
        "a red bicycle",
        style="Cinematic",
        auto_enhance=True,
    )
    assert result.startswith("a red bicycle")
    assert "cinematic" in result.lower()
    assert "highly detailed" in result.lower()


def test_build_prompt_can_leave_prompt_unchanged():
    assert (
        build_prompt(
            "minimal logo concept",
            style="None",
            auto_enhance=False,
        )
        == "minimal logo concept"
    )
