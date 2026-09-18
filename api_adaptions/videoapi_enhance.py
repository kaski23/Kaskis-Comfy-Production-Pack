"""KASKI Video Enhance API node."""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import IO

from comfy_api_nodes.nodes_hitpaw import HitPawVideoEnhance
from comfy_api_nodes.nodes_topaz import TopazVideoEnhanceV2

from .videoapi_common import (
    CATEGORY,
    Variant,
    _black_video,
    _build_core_params,
    _dynamic_variants,
    _first_output,
    _hidden_inputs,
    _model_selector,
    _run_core,
)


log = logging.getLogger(__name__)

PROMPT_TOOLTIP = (
    "Enhancement prompt. Used by Topaz Astra 2; ignored by models without "
    "prompt support."
)

VIDEO_TOOLTIP = (
    "Source video passed directly to the selected HitPaw or Topaz enhancer."
)

MODEL_TOOLTIP = (
    "HitPaw provides restoration/enhancement models. Topaz provides "
    "Astra/Starlight upscaling and optional interpolation."
)


def _topaz_label(model_name: str) -> str:
    """Expose Topaz' disabled upscaler as interpolation-only mode."""
    if model_name == "Disabled":
        return "Topaz · Interpolation Only"

    return f"Topaz · {model_name}"


ENHANCE_VARIANTS = tuple([
    *_dynamic_variants(
        HitPawVideoEnhance,
        selector_id="model",
        exclude={"video"},
        route="hitpaw_enhance",
        label=lambda name: f"HitPaw · {name}",
    ),

    *_dynamic_variants(
        TopazVideoEnhanceV2,
        selector_id="upscaler_model",
        exclude={"video", "prompt"},
        route="topaz_enhance",
        label=_topaz_label,
    ),
])

ENHANCE_LOOKUP = {
    variant.label: variant
    for variant in ENHANCE_VARIANTS
}


def _inject_inputs(
    variant: Variant,
    params: dict[str, Any],
    *,
    prompt: str,
    video: Any,
) -> None:
    """Map shared enhancement inputs to the selected provider."""
    params["video"] = video

    if (
        variant.route == "topaz_enhance"
        and variant.selector_value == "Astra 2"
    ):
        params[variant.selector_id]["prompt"] = prompt


class KASKIVideoEnhanceAPI(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VideoEnhanceAPI_KASKI",
            display_name="KASKI Video Enhance API",
            category=CATEGORY,
            description=(
                "Unified enhancement/upscaling wrapper over HitPaw and Topaz."
            ),

            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip=PROMPT_TOOLTIP,
                ),
                IO.Video.Input(
                    "video",
                    tooltip=VIDEO_TOOLTIP,
                ),
                _model_selector(
                    ENHANCE_VARIANTS,
                    MODEL_TOOLTIP,
                ),
            ],

            outputs=[
                IO.Video.Output(display_name="video"),
                IO.String.Output(display_name="prompt"),
                IO.String.Output(display_name="modelName"),
                IO.String.Output(display_name="error"),
            ],

            hidden=_hidden_inputs(),
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        video,
        model: dict[str, Any],
    ) -> IO.NodeOutput:
        model_label = ""

        try:
            model_label = model["model"]
            variant = ENHANCE_LOOKUP[model_label]

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
                video=video,
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
                "NO ERRORS TODAY - SUCCEEDED",
            )

        except Exception as error:
            log.warning(
                "[KASKI Video Enhance] %s failed: %s: %s",
                model_label or "unknown model",
                type(error).__name__,
                error,
                exc_info=True,
            )

            return IO.NodeOutput(
                _black_video(),
                prompt,
                model_label,
                f"{type(error).__name__}: {error}",
            )
