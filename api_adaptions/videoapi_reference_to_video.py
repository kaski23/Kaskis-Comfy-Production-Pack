"""KASKI Reference to Video API node."""

from __future__ import annotations

import logging
from typing import Any

import torch

from comfy_api.latest import IO

from comfy_api_nodes.nodes_bytedance import ByteDance2ReferenceNodeV2
from comfy_api_nodes.nodes_kling import (
    OmniProImageToVideoNode,
    OmniProVideoToVideoNode,
)
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
    "Shared prompt passed to the selected provider."
)

SEED_TOOLTIP = (
    "Shared execution seed. API providers may remain non-deterministic even "
    "when the seed is fixed."
)

MODEL_TOOLTIP = (
    "Seedance supports image/video/audio references. Gemini supports "
    "image/video references. Kling supports image-reference or "
    "video-reference routes, but no audio references."
)

REFERENCE_IMAGES_TOOLTIP = (
    "Image references. Seedance 2.5: up to 30; Seedance 2.0: 9; "
    "Gemini Omni: 14; Kling: 7 in image mode / 4 with a video. "
    "One image per socket; IMAGE batches are rejected."
)

REFERENCE_VIDEOS_TOOLTIP = (
    "Video references. Seedance 2.5: up to 10; Seedance 2.0: 3; "
    "Gemini Omni: 3; Kling video-reference mode: exactly 1."
)

REFERENCE_AUDIOS_TOOLTIP = (
    "Audio references. Seedance 2.5: up to 10; Seedance 2.0: 3. "
    "Gemini Omni and Kling do not accept audio references."
)


def _kling_label(model_name: str) -> str:
    """Convert Kling API model IDs into readable UI labels."""
    return {
        "kling-v3-omni": "Kling 3.0 Omni",
        "kling-video-o1": "Kling Video O1",
    }.get(model_name, model_name)


def _reference_images_input():
    return IO.Autogrow.Input(
        "reference_images",
        template=IO.Autogrow.TemplateNames(
            IO.Image.Input("reference_image"),
            names=[
                f"image_{index}"
                for index in range(1, 31)
            ],
            min=0,
        ),
        tooltip=REFERENCE_IMAGES_TOOLTIP,
    )


def _reference_videos_input():
    return IO.Autogrow.Input(
        "reference_videos",
        template=IO.Autogrow.TemplateNames(
            IO.Video.Input("reference_video"),
            names=[
                f"video_{index}"
                for index in range(1, 11)
            ],
            min=0,
        ),
        tooltip=REFERENCE_VIDEOS_TOOLTIP,
    )


def _reference_audios_input():
    return IO.Autogrow.Input(
        "reference_audios",
        template=IO.Autogrow.TemplateNames(
            IO.Audio.Input("reference_audio"),
            names=[
                f"audio_{index}"
                for index in range(1, 11)
            ],
            min=0,
        ),
        tooltip=REFERENCE_AUDIOS_TOOLTIP,
    )


def _clean_group(
    group: IO.Autogrow.Type | None,
) -> dict[str, Any]:
    """Remove unused reference sockets while preserving order."""
    if not group:
        return {}

    return {
        str(key): value
        for key, value in group.items()
        if value is not None
    }


def _validate_reference_images(
    images: dict[str, Any],
) -> None:
    """Enforce exactly one logical image per reference socket."""
    for name, image in images.items():
        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"{name}: expected a Comfy IMAGE tensor."
            )

        if image.ndim != 4:
            raise ValueError(
                f"{name}: expected BHWC IMAGE tensor, "
                f"got {tuple(image.shape)}."
            )

        if image.shape[0] != 1:
            raise ValueError(
                f"{name}: IMAGE batches are not supported in one reference "
                f"socket. Received batch size {image.shape[0]}. "
                "Connect each image to its own reference socket."
            )


def _images_to_batch(
    images: dict[str, Any],
) -> torch.Tensor | None:
    """Convert separate image sockets into the IMAGE batch Kling expects."""
    if not images:
        return None

    tensors = list(images.values())
    shapes = {
        tuple(tensor.shape[1:])
        for tensor in tensors
    }

    if len(shapes) != 1:
        raise ValueError(
            "Kling receives reference images as one IMAGE batch. "
            "All connected Kling reference images must have matching dimensions."
        )

    return torch.cat(
        tensors,
        dim=0,
    )


REFERENCE_VARIANTS = tuple([
    *_dynamic_variants(
        ByteDance2ReferenceNodeV2,
        selector_id="model",
        exclude=COMMON_EXCLUDE
        | {
            "reference_images",
            "reference_videos",
            "reference_audios",
        },
        route="seedance_reference",
        label=lambda name: name,
    ),

    *_combo_variants(
        OmniProImageToVideoNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE
        | {
            "reference_images",
        },
        route="kling_image_reference",
        label=lambda name: f"{_kling_label(name)} · Image References",
    ),

    *_combo_variants(
        OmniProVideoToVideoNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE
        | {
            "reference_video",
            "reference_images",
        },
        route="kling_video_reference",
        label=lambda name: f"{_kling_label(name)} · Video Reference",
    ),

    *_dynamic_variants(
        GeminiVideoOmni,
        selector_id="model",
        exclude=COMMON_EXCLUDE
        | {
            "images",
            "videos",
        },
        route="gemini_reference",
        label=lambda name: f"Google {name}",
    ),
])

REFERENCE_LOOKUP = {
    variant.label: variant
    for variant in REFERENCE_VARIANTS
}


def _inject_inputs(
    variant: Variant,
    params: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    reference_images: dict[str, Any],
    reference_videos: dict[str, Any],
    reference_audios: dict[str, Any],
) -> None:
    """Map shared reference inputs to the selected provider."""
    if variant.route.startswith("seedance_") or variant.route.startswith("gemini_"):
        params[variant.selector_id]["prompt"] = prompt
    elif variant.route.startswith("kling_"):
        params["prompt"] = prompt

    params["seed"] = seed

    if variant.route == "seedance_reference":
        model = params[variant.selector_id]
        model["reference_images"] = reference_images
        model["reference_videos"] = reference_videos
        model["reference_audios"] = reference_audios

    elif variant.route == "gemini_reference":
        if reference_audios:
            raise ValueError(
                "Google Gemini Omni does not support audio references."
            )

        model = params[variant.selector_id]
        model["images"] = reference_images
        model["videos"] = reference_videos

    elif variant.route == "kling_image_reference":
        if reference_videos:
            raise ValueError(
                "Kling Image References does not accept video references. "
                "Select the Kling Video Reference model instead."
            )

        if reference_audios:
            raise ValueError(
                "Kling Omni does not support audio references."
            )

        image_batch = _images_to_batch(reference_images)

        if image_batch is None:
            raise ValueError(
                "Kling Image References requires at least one image."
            )

        params["reference_images"] = image_batch

    elif variant.route == "kling_video_reference":
        if reference_audios:
            raise ValueError(
                "Kling Omni does not support audio references."
            )

        if len(reference_videos) != 1:
            raise ValueError(
                "Kling Video Reference requires exactly one reference video."
            )

        params["reference_video"] = next(
            iter(reference_videos.values())
        )

        image_batch = _images_to_batch(reference_images)

        if image_batch is not None:
            params["reference_images"] = image_batch


class KASKIReferenceToVideoAPI(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ReferenceToVideoAPI_KASKI",
            display_name="KASKI Reference to Video API",
            category=CATEGORY,
            description=(
                "Unified reference-to-video wrapper over Seedance, Kling Omni "
                "and Gemini Omni."
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
                _reference_images_input(),
                _reference_videos_input(),
                _reference_audios_input(),
                _model_selector(
                    REFERENCE_VARIANTS,
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
        reference_images: IO.Autogrow.Type | None,
        reference_videos: IO.Autogrow.Type | None,
        reference_audios: IO.Autogrow.Type | None,
        model: dict[str, Any],
    ) -> IO.NodeOutput:
        model_label = ""

        try:
            model_label = model["model"]
            variant = REFERENCE_LOOKUP[model_label]

            settings = dict(model)
            settings.pop("model", None)

            clean_images = _clean_group(reference_images)
            clean_videos = _clean_group(reference_videos)
            clean_audios = _clean_group(reference_audios)

            _validate_reference_images(clean_images)

            params = _build_core_params(
                variant,
                settings,
            )

            _inject_inputs(
                variant,
                params,
                prompt=prompt,
                seed=seed,
                reference_images=clean_images,
                reference_videos=clean_videos,
                reference_audios=clean_audios,
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
                "[KASKI Reference to Video] %s failed: %s: %s",
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
