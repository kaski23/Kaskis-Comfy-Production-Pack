"""KASKI Unified Video API.

A thin two-stage wrapper around ComfyUI's built-in video API nodes.

The node intentionally does not reimplement provider APIs. It:
1. imports ComfyUI's original provider nodes,
2. reuses their model-specific input definitions,
3. keeps prompt / seed and mode-specific reference sockets stable,
4. reconstructs the selected provider's original execute() arguments,
5. runs the original ComfyUI node,
6. soft-fails to one second of black video if the provider call errors.

Provider-side validation, uploads, authentication, API requests, polling and
downloads remain owned by ComfyUI.
"""

from __future__ import annotations

import copy
import inspect
import logging
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

import torch

from comfy_api.latest import IO, Input, InputImpl, Types

from comfy_api_nodes.nodes_bytedance import (
    ByteDance2TextToVideoNode,
    ByteDance2FirstLastFrameNode,
    ByteDance2ReferenceNodeV2,
)

from comfy_api_nodes.nodes_kling import (
    OmniProTextToVideoNode,
    OmniProFirstLastFrameNode,
    OmniProImageToVideoNode,
    OmniProVideoToVideoNode,
)

from comfy_api_nodes.nodes_gemini import GeminiVideoOmni


log = logging.getLogger(__name__)

CATEGORY = "KASKI/api-adaptions/video"

MODE_TEXT = "TEXT_TO_VIDEO"
MODE_FIRST_LAST = "FIRST_LAST_FRAME"
MODE_REFERENCE = "REFERENCE_VIDEO"


# ---------------------------------------------------------------------------
# Capability tooltips
# ---------------------------------------------------------------------------

MODE_TOOLTIP = (
    "Choose the generation structure first. Switching models inside the same "
    "mode keeps the shared prompt/reference sockets connected."
)

MODEL_TOOLTIPS = {
    MODE_TEXT: (
        "Seedance 2.5: newest, up to 30s / 720p. Seedance 2.0: up to 4K / 15s; "
        "Fast/Mini prioritize speed. Kling 3.0 Omni supports storyboard/audio; "
        "Video O1 has fewer features. Gemini Omni generates video with audio."
    ),
    MODE_FIRST_LAST: (
        "Seedance 2.5/2.0 and Kling Omni support first + optional last frame. "
        "Kling storyboard/audio features depend on the selected Kling model."
    ),
    MODE_REFERENCE: (
        "Seedance supports image/video/audio references. Gemini supports image/video. "
        "Kling supports image-reference or video-reference routes, but no audio refs."
    ),
}

PROMPT_TOOLTIP = (
    "Shared prompt passed to the selected provider. Provider-specific prompt "
    "rules and limits still apply."
)

SEED_TOOLTIP = (
    "Shared execution seed. API providers may remain non-deterministic even "
    "when the seed is fixed."
)

FIRST_FRAME_TOOLTIP = (
    "Start frame. Used by Seedance First/Last and Kling Omni First/Last."
)

LAST_FRAME_TOOLTIP = (
    "Optional end frame. Used by Seedance First/Last and Kling Omni First/Last."
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

FALLBACK_PROVIDER_TOOLTIP = "Provider-specific setting inherited from ComfyUI."


# ---------------------------------------------------------------------------
# Variant description
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """One selectable provider/model implementation inside a generation mode.

    selector_id / selector_value describe the model selector expected by the
    original ComfyUI node.

    Dynamic selectors (Seedance/Gemini) need their model-specific settings
    rebuilt as a nested dictionary. Ordinary selectors (Kling) stay flat.
    """

    label: str
    node_cls: type[IO.ComfyNode]

    selector_id: str
    selector_value: str
    selector_is_dynamic: bool

    nested_ids: frozenset[str]
    inputs: tuple[Input, ...]

    route: str


# ---------------------------------------------------------------------------
# Core-schema helpers
# ---------------------------------------------------------------------------

def _schema_input(
    node_cls: type[IO.ComfyNode],
    input_id: str,
) -> Input:
    """Find one input in an original ComfyUI node schema."""
    for item in node_cls.define_schema().inputs:
        if item.id == input_id:
            return item

    raise RuntimeError(
        f"{node_cls.__name__} no longer exposes input {input_id!r}."
    )


def _copy_input(item: Input) -> Input:
    """Clone a ComfyUI input so KASKI can safely add missing tooltip text."""
    cloned = copy.copy(item)

    if not getattr(cloned, "tooltip", None):
        cloned.tooltip = FALLBACK_PROVIDER_TOOLTIP

    return cloned


def _clone_filtered(
    inputs: list[Input],
    exclude: set[str],
) -> list[Input]:
    """Copy provider inputs while removing sockets owned by KASKI."""
    return [
        _copy_input(item)
        for item in inputs
        if item.id not in exclude
    ]


def _dynamic_variants(
    node_cls: type[IO.ComfyNode],
    *,
    selector_id: str,
    exclude: set[str],
    route: str,
    label: Callable[[str], str],
) -> list[Variant]:
    """Expand a Comfy DynamicCombo model selector into KASKI model choices.

    Seedance and Gemini store model-specific controls inside a DynamicCombo.
    Each option becomes one entry in KASKI's second-stage model selector.
    """
    schema = node_cls.define_schema()
    selector = _schema_input(node_cls, selector_id)

    if not isinstance(selector, IO.DynamicCombo.Input):
        raise TypeError(
            f"{node_cls.__name__}.{selector_id} is no longer a DynamicCombo."
        )

    # Inputs outside the provider's model selector, e.g. Seedance watermark.
    common_inputs = _clone_filtered(
        [item for item in schema.inputs if item.id != selector_id],
        exclude,
    )

    variants: list[Variant] = []

    for option in selector.options:
        nested_inputs = _clone_filtered(
            option.inputs,
            exclude,
        )

        variants.append(
            Variant(
                label=label(option.key),
                node_cls=node_cls,
                selector_id=selector_id,
                selector_value=option.key,
                selector_is_dynamic=True,
                nested_ids=frozenset(
                    item.id
                    for item in nested_inputs
                ),
                inputs=tuple(
                    nested_inputs
                    + [_copy_input(item) for item in common_inputs]
                ),
                route=route,
            )
        )

    return variants


def _combo_variants(
    node_cls: type[IO.ComfyNode],
    *,
    selector_id: str,
    exclude: set[str],
    route: str,
    label: Callable[[str], str],
) -> list[Variant]:
    """Expand a normal Comfy Combo model selector into KASKI model choices.

    Kling exposes model_name as a normal Combo, so every model value becomes
    one KASKI model option while the remaining Kling settings stay flat.
    """
    schema = node_cls.define_schema()
    selector = _schema_input(node_cls, selector_id)

    if not isinstance(selector, IO.Combo.Input):
        raise TypeError(
            f"{node_cls.__name__}.{selector_id} is no longer a Combo."
        )

    common_inputs = _clone_filtered(
        [item for item in schema.inputs if item.id != selector_id],
        exclude,
    )

    return [
        Variant(
            label=label(str(model_name)),
            node_cls=node_cls,
            selector_id=selector_id,
            selector_value=str(model_name),
            selector_is_dynamic=False,
            nested_ids=frozenset(),
            inputs=tuple(
                _copy_input(item)
                for item in common_inputs
            ),
            route=route,
        )
        for model_name in selector.options
    ]


def _kling_label(model_name: str) -> str:
    """Convert Kling's API model IDs into readable UI labels."""
    return {
        "kling-v3-omni": "Kling 3.0 Omni",
        "kling-video-o1": "Kling Video O1",
    }.get(model_name, model_name)


# ---------------------------------------------------------------------------
# Build model registries directly from ComfyUI's current provider schemas
# ---------------------------------------------------------------------------

# KASKI owns these inputs globally or at generation-mode level.
COMMON_EXCLUDE = {
    "prompt",
    "seed",
}


TEXT_VARIANTS = [
    # Seedance model-specific controls come from the original DynamicCombo.
    *_dynamic_variants(
        ByteDance2TextToVideoNode,
        selector_id="model",
        exclude=COMMON_EXCLUDE,
        route="seedance_text",
        label=lambda name: name,
    ),

    # Kling exposes its model choice as model_name.
    *_combo_variants(
        OmniProTextToVideoNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE,
        route="kling_text",
        label=_kling_label,
    ),

    # Gemini Omni uses the same core node for text and reference generation.
    *_dynamic_variants(
        GeminiVideoOmni,
        selector_id="model",
        exclude=COMMON_EXCLUDE | {"images", "videos"},
        route="gemini_text",
        label=lambda name: f"Google {name}",
    ),
]


FIRST_LAST_VARIANTS = [
    *_dynamic_variants(
        ByteDance2FirstLastFrameNode,
        selector_id="model",
        exclude=COMMON_EXCLUDE
        | {
            "first_frame",
            "last_frame",
        },
        route="seedance_first_last",
        label=lambda name: name,
    ),

    *_combo_variants(
        OmniProFirstLastFrameNode,
        selector_id="model_name",
        exclude=COMMON_EXCLUDE
        | {
            "first_frame",
            "end_frame",
        },
        route="kling_first_last",
        label=_kling_label,
    ),
]


REFERENCE_VARIANTS = [
    # Seedance accepts all three reference media types.
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

    # Kling separates image-reference and video-reference generation.
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

    # Gemini Omni accepts images and videos but no audio references.
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
]


MODE_VARIANTS: dict[str, tuple[Variant, ...]] = {
    MODE_TEXT: tuple(TEXT_VARIANTS),
    MODE_FIRST_LAST: tuple(FIRST_LAST_VARIANTS),
    MODE_REFERENCE: tuple(REFERENCE_VARIANTS),
}


VARIANT_LOOKUP: dict[str, dict[str, Variant]] = {
    mode: {
        variant.label: variant
        for variant in variants
    }
    for mode, variants in MODE_VARIANTS.items()
}


# ---------------------------------------------------------------------------
# KASKI UI
# ---------------------------------------------------------------------------

def _model_selector(mode: str) -> Input:
    """Second-stage selector: model + only that model's provider settings."""
    return IO.DynamicCombo.Input(
        "model",
        options=[
            IO.DynamicCombo.Option(
                variant.label,
                [
                    _copy_input(item)
                    for item in variant.inputs
                ],
            )
            for variant in MODE_VARIANTS[mode]
        ],
        tooltip=MODEL_TOOLTIPS[mode],
    )


def _reference_images_input() -> Input:
    """Stable image-reference sockets sized to Seedance 2.5's maximum."""
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


def _reference_videos_input() -> Input:
    """Stable video-reference sockets sized to Seedance 2.5's maximum."""
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


def _reference_audios_input() -> Input:
    """Stable audio-reference sockets sized to Seedance 2.5's maximum."""
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


def _generation_mode_selector() -> Input:
    """First-stage selector.

    The selected mode owns the stable sockets. Therefore changing only the
    model inside a mode does not destroy prompt/reference connections.
    """
    return IO.DynamicCombo.Input(
        "generation_mode",
        options=[
            IO.DynamicCombo.Option(
                MODE_TEXT,
                [
                    _model_selector(MODE_TEXT),
                ],
            ),

            IO.DynamicCombo.Option(
                MODE_FIRST_LAST,
                [
                    _model_selector(MODE_FIRST_LAST),

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
                ],
            ),

            IO.DynamicCombo.Option(
                MODE_REFERENCE,
                [
                    _model_selector(MODE_REFERENCE),
                    _reference_images_input(),
                    _reference_videos_input(),
                    _reference_audios_input(),
                ],
            ),
        ],
        tooltip=MODE_TOOLTIP,
    )


# ---------------------------------------------------------------------------
# Runtime helpers
# ---------------------------------------------------------------------------

async def _run_core(
    node_cls: type[IO.ComfyNode],
    active_cls: type[IO.ComfyNode],
    **params: Any,
):
    """Run the original ComfyUI execute() implementation.

    The function body comes from the provider node, but active_cls remains the
    KASKI node. This preserves ComfyUI's hidden auth/API-key execution context.
    """
    execute_func = getattr(
        node_cls.execute,
        "__func__",
        None,
    )

    if execute_func is None:
        raise RuntimeError(
            f"{node_cls.__name__}.execute is no longer a classmethod."
        )

    result = execute_func(
        active_cls,
        **params,
    )

    if inspect.isawaitable(result):
        result = await result

    return result


def _first_output(result: Any):
    """All wrapped provider nodes return VIDEO as output 0."""
    values = (
        result.args
        if isinstance(result, IO.NodeOutput)
        else result
    )

    if not isinstance(values, (tuple, list)) or not values:
        raise TypeError(
            f"Unexpected core output type: {type(result).__name__}"
        )

    return values[0]


def _black_video(
    width: int = 512,
    height: int = 512,
    fps: int = 24,
) -> Input.Video:
    """Create a native one-second black VIDEO for soft-error fallback."""
    frames = torch.zeros(
        (fps, height, width, 3),
        dtype=torch.float32,
    )

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=frames,
            audio=None,
            frame_rate=Fraction(fps, 1),
        )
    )


def _clean_group(
    group: IO.Autogrow.Type | None,
) -> dict[str, Any]:
    """Remove unused Autogrow slots while preserving their order."""
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
    """Enforce one logical image per KASKI reference socket.

    Comfy IMAGE sockets can technically carry a batch. KASKI rejects that here
    because image_1, image_2, ... should map predictably to provider references.
    """
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
    """Convert KASKI's individual image sockets into Kling's IMAGE batch."""
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
            "All connected Kling reference images must therefore have "
            "matching dimensions."
        )

    return torch.cat(
        tensors,
        dim=0,
    )


def _build_core_params(
    variant: Variant,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the parameter structure expected by the original provider node.

    Seedance/Gemini:
        model = {
            "model": selected_model,
            ...nested model settings...
        }
        plus any original top-level settings such as watermark.

    Kling:
        model_name = selected_model
        plus its remaining flat settings.
    """
    settings = dict(settings)

    if variant.selector_is_dynamic:
        nested_model = {
            variant.selector_id: variant.selector_value,
        }

        # Move the settings that originally lived inside Comfy's DynamicCombo
        # back into that nested model dictionary.
        for key in variant.nested_ids:
            if key in settings:
                nested_model[key] = settings.pop(key)

        params = {
            variant.selector_id: nested_model,
        }
        params.update(settings)

        return params

    # Kling uses a normal Combo, so its provider settings stay flat.
    params = {
        variant.selector_id: variant.selector_value,
    }
    params.update(settings)

    return params


def _inject_stable_inputs(
    variant: Variant,
    params: dict[str, Any],
    *,
    prompt: str,
    seed: int,
    generation_mode: dict[str, Any],
) -> None:
    """Map KASKI's stable inputs back onto each provider's native interface."""
    route = variant.route

    # Seedance and Gemini keep the prompt inside their nested model dictionary.
    if route.startswith("seedance_") or route.startswith("gemini_"):
        params[variant.selector_id]["prompt"] = prompt

    # Kling expects prompt as a normal execute() argument.
    if route.startswith("kling_"):
        params["prompt"] = prompt

    # All wrapped nodes expose seed at top level.
    params["seed"] = seed

    # ------------------------------------------------------------------
    # First / last frame mapping
    # ------------------------------------------------------------------

    if route == "seedance_first_last":
        params["first_frame"] = generation_mode.get("first_frame")
        params["last_frame"] = generation_mode.get("last_frame")

    elif route == "kling_first_last":
        params["first_frame"] = generation_mode.get("first_frame")
        params["end_frame"] = generation_mode.get("last_frame")

    # ------------------------------------------------------------------
    # Reference mapping
    # ------------------------------------------------------------------

    elif route == "seedance_reference":
        model = params[variant.selector_id]

        model["reference_images"] = generation_mode["reference_images"]
        model["reference_videos"] = generation_mode["reference_videos"]
        model["reference_audios"] = generation_mode["reference_audios"]

    elif route == "gemini_reference":
        if generation_mode["reference_audios"]:
            raise ValueError(
                "Google Gemini Omni does not support audio references."
            )

        model = params[variant.selector_id]

        model["images"] = generation_mode["reference_images"]
        model["videos"] = generation_mode["reference_videos"]

    elif route == "kling_image_reference":
        if generation_mode["reference_videos"]:
            raise ValueError(
                "Kling Image References does not accept video references. "
                "Select the Kling Video Reference model instead."
            )

        if generation_mode["reference_audios"]:
            raise ValueError(
                "Kling Omni does not support audio references."
            )

        image_batch = _images_to_batch(
            generation_mode["reference_images"]
        )

        if image_batch is None:
            raise ValueError(
                "Kling Image References requires at least one image."
            )

        params["reference_images"] = image_batch

    elif route == "kling_video_reference":
        videos = generation_mode["reference_videos"]

        if len(videos) != 1:
            raise ValueError(
                "Kling Video Reference requires exactly one reference video."
            )

        if generation_mode["reference_audios"]:
            raise ValueError(
                "Kling Omni does not support audio references."
            )

        params["reference_video"] = next(
            iter(videos.values())
        )

        image_batch = _images_to_batch(
            generation_mode["reference_images"]
        )

        if image_batch is not None:
            params["reference_images"] = image_batch


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class KASKIVideoAPI(IO.ComfyNode):
    """Stable KASKI UI in front of ComfyUI's original provider nodes."""

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="VideoAPI_KASKI",
            display_name="KASKI Unified Video API",
            category=CATEGORY,
            description=(
                "Unified Seedance, Kling Omni and Gemini Omni video wrapper. "
                "Provider execution stays inside ComfyUI; KASKI provides a "
                "stable two-stage UI and soft error handling."
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

                _generation_mode_selector(),
            ],

            outputs=[
                IO.Video.Output(
                    display_name="video",
                    tooltip=(
                        "Generated provider video, or a one-second black "
                        "placeholder when generation fails."
                    ),
                ),

                IO.String.Output(
                    display_name="prompt",
                    tooltip="The prompt passed into this generation.",
                ),

                IO.String.Output(
                    display_name="modelName",
                    tooltip="Selected KASKI provider/model route.",
                ),

                IO.Int.Output(
                    display_name="seed",
                    tooltip="Seed passed into this generation.",
                ),

                IO.String.Output(
                    display_name="error",
                    tooltip=(
                        "Empty on success. Contains the caught exception on "
                        "soft failure."
                    ),
                ),
            ],

            # These hidden inputs are supplied by ComfyUI at execution time.
            # _run_core() deliberately executes provider code with this KASKI
            # class as cls so Comfy's auth/upload helpers keep this context.
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],

            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int,
        generation_mode: dict[str, Any],
    ) -> IO.NodeOutput:
        """Resolve the selected route, rebuild provider args and run it.

        Any provider/API exception is caught here so one failed generation does
        not abort other parallel branches of the ComfyUI workflow.
        """
        mode = ""
        model_label = ""

        try:
            # --------------------------------------------------------------
            # 1. Resolve outer mode and inner model
            # --------------------------------------------------------------

            mode = generation_mode["generation_mode"]

            model_block = generation_mode["model"]
            model_label = model_block["model"]

            variant = VARIANT_LOOKUP[mode][model_label]

            # Everything below model_block["model"] is provider-specific
            # configuration copied from the original Comfy node.
            settings = dict(model_block)
            settings.pop("model", None)

            # --------------------------------------------------------------
            # 2. Normalize stable reference sockets
            # --------------------------------------------------------------

            if mode == MODE_REFERENCE:
                generation_mode = dict(generation_mode)

                generation_mode["reference_images"] = _clean_group(
                    generation_mode.get("reference_images")
                )

                generation_mode["reference_videos"] = _clean_group(
                    generation_mode.get("reference_videos")
                )

                generation_mode["reference_audios"] = _clean_group(
                    generation_mode.get("reference_audios")
                )

                _validate_reference_images(
                    generation_mode["reference_images"]
                )

            # --------------------------------------------------------------
            # 3. Reconstruct the original provider node arguments
            # --------------------------------------------------------------

            params = _build_core_params(
                variant,
                settings,
            )

            _inject_stable_inputs(
                variant,
                params,
                prompt=prompt,
                seed=seed,
                generation_mode=generation_mode,
            )

            # --------------------------------------------------------------
            # 4. Delegate the actual API work back to ComfyUI
            # --------------------------------------------------------------

            result = await _run_core(
                variant.node_cls,
                cls,
                **params,
            )

            # Wrapped nodes all expose VIDEO as output 0.
            video = _first_output(result)

            return IO.NodeOutput(
                video,
                prompt,
                model_label,
                seed,
                "NO ERRORS - SUCCESSFULLY GENERATED",
            )

        except Exception as error:
            # --------------------------------------------------------------
            # Soft failure: log full traceback, keep the workflow alive
            # --------------------------------------------------------------

            log.warning(
                "[KASKI Video API] %s / %s failed: %s: %s",
                mode or "unknown mode",
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


UNIFIED_VIDEOAPI_NODES_LIST = [
    KASKIVideoAPI,
]
