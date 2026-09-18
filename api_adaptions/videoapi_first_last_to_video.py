"""KASKI First / Last Frame to Video API node."""

from __future__ import annotations

import logging
from typing import Any

from comfy_api.latest import IO

from comfy_api_nodes.nodes_bytedance import ByteDance2FirstLastFrameNode
from comfy_api_nodes.nodes_kling import OmniProFirstLastFrameNode

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
    "Shared prompt passed to the selected provider."
)

SEED_TOOLTIP = (
    "Shared execution seed. API providers may remain non-deterministic even "
    "when the seed is fixed."
)

FIRST_FRAME_TOOLTIP = (
    "Start frame used by the selected provider."
)

LAST_FRAME_TOOLTIP = (
    "Optional end frame used as the target state of the motion."
)

MODEL_TOOLTIP = (
    "Seedance and Kling Omni support first/last-frame generation."
)


def _kling_label(model_name: str) -> str:
    """Convert Kling API model IDs into readable UI labels."""
    return {
        "kling-v3-omni": "Kling 3.0 Omni",
        "kling-video-o1": "Kling Video O1",
    }.get(model_name, model_name)


FIRST_LAST_VARIANTS = tuple([
    *_dynamic_variants(
        ByteDance2FirstLastFrameNode,
        selector_id="model",
        exclude=COMMON_EXCLUDE | {"first_frame", "last_frame"},
        route="seedance_first_last",
        label=lambda name: name,
    ),
    *_combo_variants(
        OmniProFirstLastFrameNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE | {"first_frame", "end_frame"},
        route="kling_first_last",
        label=_kling_label,
    ),
])

FIRST_LAST_LOOKUP = {
    variant.label: variant
    for variant in FIRST_LAST_VARIANTS
}


def _inject_inputs(
    variant: Variant,
    params: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    first_frame: Any,
    last_frame: Any,
) -> None:
    """Map shared first/last-frame inputs to the selected provider."""
    if variant.route.startswith("seedance_"):
        params[variant.selector_id]["prompt"] = prompt
    elif variant.route.startswith("kling_"):
        params["prompt"] = prompt

    params["seed"] = seed

    if variant.route == "seedance_first_last":
        params["first_frame"] = first_frame
        params["last_frame"] = last_frame

    elif variant.route == "kling_first_last":
        params["first_frame"] = first_frame
        params["end_frame"] = last_frame


class KASKIFirstLastFrameToVideoAPI(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FirstLastFrameToVideoAPI_KASKI",
            display_name="KASKI First / Last Frame to Video API",
            category=CATEGORY,
            description=(
                "Unified first/last-frame wrapper over Seedance and Kling Omni."
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
                IO.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip=FIRST_FRAME_TOOLTIP,
                ),
                IO.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip=LAST_FRAME_TOOLTIP,
                ),
                _model_selector(
                    FIRST_LAST_VARIANTS,
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
        first_frame,
        last_frame,
        model: dict[str, Any],
    ) -> IO.NodeOutput:
        model_label = ""

        try:
            model_label = model["model"]
            variant = FIRST_LAST_LOOKUP[model_label]

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
                first_frame=first_frame,
                last_frame=last_frame,
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
                "[KASKI First/Last to Video] %s failed: %s: %s",
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
