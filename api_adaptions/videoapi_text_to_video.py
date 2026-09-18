"""KASKI Text to Video API node."""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import IO

from comfy_api_nodes.nodes_bytedance import ByteDance2TextToVideoNode
from comfy_api_nodes.nodes_kling import OmniProTextToVideoNode
from comfy_api_nodes.nodes_gemini import GeminiVideoOmni

from .videoapi_common import (
    CATEGORY,
    Variant,
    _black_video,
    _build_core_params,
    _combo_variants,
    _dynamic_variants,
    _first_output,
    _hidden_inputs,
    _model_selector,
    _run_core,
)


log = logging.getLogger(__name__)

COMMON_EXCLUDE = {"prompt", "seed"}

PROMPT_TOOLTIP = (
    "Shared prompt passed to the selected provider. Provider-specific prompt "
    "rules and limits still apply."
)

SEED_TOOLTIP = (
    "Shared execution seed. API providers may remain non-deterministic even "
    "when the seed is fixed."
)

MODEL_TOOLTIP = (
    "Seedance, Kling Omni and Gemini Omni all support text-to-video."
)


def _kling_label(model_name: str) -> str:
    """Convert Kling API model IDs into readable UI labels."""
    return {
        "kling-v3-omni": "Kling 3.0 Omni",
        "kling-video-o1": "Kling Video O1",
    }.get(model_name, model_name)


TEXT_VARIANTS = tuple([
    *_dynamic_variants(
        ByteDance2TextToVideoNode,
        selector_id="model",
        exclude=COMMON_EXCLUDE,
        route="seedance_text",
        label=lambda name: name,
    ),
    *_combo_variants(
        OmniProTextToVideoNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE,
        route="kling_text",
        label=_kling_label,
    ),
    *_dynamic_variants(
        GeminiVideoOmni,
        selector_id="model",
        exclude=COMMON_EXCLUDE | {"images", "videos"},
        route="gemini_text",
        label=lambda name: f"Google {name}",
    ),
])

TEXT_LOOKUP = {
    variant.label: variant
    for variant in TEXT_VARIANTS
}


def _inject_inputs(
    variant: Variant,
    params: dict[str, Any],
    *,
    prompt: str,
    seed: int,
) -> None:
    """Map the shared KASKI inputs to the selected provider."""
    if variant.route.startswith("seedance_") or variant.route.startswith("gemini_"):
        params[variant.selector_id]["prompt"] = prompt
    elif variant.route.startswith("kling_"):
        params["prompt"] = prompt

    params["seed"] = seed


class KASKITextToVideoAPI(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TextToVideoAPI_KASKI",
            display_name="KASKI Text to Video API",
            category=CATEGORY,
            description=(
                "Unified text-to-video wrapper over Seedance, Kling Omni and "
                "Gemini Omni."
            ),

            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip=PROMPT_TOOLTIP,
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=SEED_TOOLTIP,
                ),
                _model_selector(
                    TEXT_VARIANTS,
                    MODEL_TOOLTIP,
                ),
            ],

            outputs=[
                IO.Video.Output(display_name="video"),
                IO.String.Output(display_name="prompt"),
                IO.String.Output(display_name="modelName"),
                IO.Int.Output(display_name="seed"),
                IO.String.Output(display_name="error"),
            ],

            hidden=_hidden_inputs(),
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int,
        model: dict[str, Any],
    ) -> IO.NodeOutput:
        model_label = ""

        try:
            model_label = model["model"]
            variant = TEXT_LOOKUP[model_label]

            settings = dict(model)
            settings.pop("model", None)

            params = _build_core_params(
                variant,
                settings,
            )

            _inject_inputs(
                variant,
                params,
                prompt=prompt,
                seed=seed,
            )

            result = await _run_core(
                variant.node_cls,
                cls,
                **params,
            )

            return IO.NodeOutput(
                _first_output(result),
                prompt,
                model_label,
                seed,
                "NO ERRORS TODAY - SUCCEEDED",
            )

        except Exception as error:
            log.warning(
                "[KASKI Text to Video] %s failed: %s: %s",
                model_label or "unknown model",
                type(error).__name__,
                error,
                exc_info=True,
            )

            return IO.NodeOutput(
                _black_video(),
                prompt,
                model_label,
                seed,
                f"{type(error).__name__}: {error}",
            )
