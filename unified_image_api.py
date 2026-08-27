from __future__ import annotations

import traceback
from typing import Any

import torch

from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.bytedance import (
    RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
)
from comfy_api_nodes.nodes_bfl import Flux2ImageNode, _flux2_model_inputs
from comfy_api_nodes.nodes_bytedance import ByteDanceSeedreamNodeV2, _seedream_model_inputs
from comfy_api_nodes.nodes_gemini import (
    GEMINI_IMAGE_SYS_PROMPT,
    GeminiImage2,
    GeminiNanoBanana2V2,
    _nano_banana_2_v2_model_inputs,
)
from comfy_api_nodes.nodes_openai import (
    OpenAIGPTImageNodeV2,
    _gpt_image_legacy_model_inputs,
    _gpt_image_shared_inputs,
)


CATEGORY = "KASKI/api-adaptions/image"
SETTINGS_TYPE = "KASKI_IMAGE_API_SETTINGS"

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_SEEDREAM = "seedream"
PROVIDER_FLUX2 = "flux2"

OPENAI_LABEL = "OpenAI GPT Image"
GEMINI_LABEL = "Gemini / Nanobanana"
SEEDREAM_LABEL = "ByteDance Seedream"
FLUX2_LABEL = "Black Forest Labs FLUX.2"

GEMINI_PRO = "Gemini 3 Pro Image"
GEMINI_NB2 = "Nano Banana 2 (Gemini 3.1 Flash Image)"
GEMINI_NB2_LITE = "Nano Banana 2 Lite"

GEMINI_BASE_RATIOS = [
    "auto", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9",
]


# -----------------------------------------------------------------------------
# Generic glue
# -----------------------------------------------------------------------------


def _black_image(width: int = 1024, height: int = 1024) -> torch.Tensor:
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _black_like(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim < 3:
        return _black_image()
    h, w = (
        (int(image.shape[1]), int(image.shape[2]))
        if image.ndim == 4
        else (int(image.shape[0]), int(image.shape[1]))
    )
    dtype = image.dtype if image.is_floating_point() else torch.float32
    return torch.zeros((1, h, w, 3), dtype=dtype, device=image.device)


def _log_soft_error(where: str, error: Exception) -> None:
    print(f"[KASKI Image API] {where}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _image_group(images: Input.Image | None) -> dict[str, Input.Image]:
    # Keep KASKI's one IMAGE socket. Core nodes flatten/count BHWC batches.
    return {} if images is None else {"image_1": images}


def _without(inputs: list[Input], *ids: str) -> list[Input]:
    """Reuse Comfy widget definitions while removing inputs owned by Generator."""
    blocked = set(ids)
    return [item for item in inputs if getattr(item, "id", None) not in blocked]


async def _run_core(
    core_node: type[IO.ComfyNode],
    active_cls: type[IO.ComfyNode],
    **kwargs: Any,
) -> IO.NodeOutput:
    """
    Execute Comfy's original classmethod body with KASKI's active node class.

    Comfy's API helpers receive `cls`; injecting the active KASKI class keeps the
    hidden auth/API context from the node that is actually running in the graph.
    """
    func = getattr(core_node.execute, "__func__", None)
    if func is None:
        raise RuntimeError(
            f"{core_node.__name__}.execute is no longer a classmethod."
        )
    return await func(active_cls, **kwargs)


# -----------------------------------------------------------------------------
# Settings UI
# -----------------------------------------------------------------------------


def _openai_selector() -> Input:
    quality_only = _without(_gpt_image_shared_inputs(), "images", "mask")

    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option(
                "gpt-image-2",
                [
                    IO.Combo.Input(
                        "size",
                        default="auto",
                        options=[
                            "auto", "1024x1024", "1024x1536", "1536x1024",
                            "2048x2048", "2048x1152", "1152x2048",
                            "3840x2160", "2160x3840", "Custom",
                        ],
                    ),
                    IO.Int.Input(
                        "custom_width", default=1024, min=1024, max=3840, step=16,
                    ),
                    IO.Int.Input(
                        "custom_height", default=1024, min=1024, max=3840, step=16,
                    ),
                    IO.Combo.Input(
                        "background", default="auto", options=["auto", "opaque"],
                    ),
                    *quality_only,
                ],
            ),
            IO.DynamicCombo.Option(
                "gpt-image-1.5",
                _without(_gpt_image_legacy_model_inputs(), "images", "mask"),
            ),
            IO.DynamicCombo.Option(
                "gpt-image-1",
                _without(_gpt_image_legacy_model_inputs(), "images", "mask"),
            ),
        ],
    )


def _gemini_selector() -> Input:
    nb2 = _without(
        _nano_banana_2_v2_model_inputs(["1K", "2K", "4K"]),
        "images",
        "files",
    )
    nb2_lite = _without(
        _nano_banana_2_v2_model_inputs(["1K"]),
        "images",
        "files",
    )

    sampling = lambda: [
        IO.Float.Input(
            "temperature", default=1.0, min=0.0, max=2.0, step=0.01,
            advanced=True,
        ),
        IO.Float.Input(
            "top_p", default=0.95, min=0.0, max=1.0, step=0.01,
            advanced=True,
        ),
    ]

    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option(
                GEMINI_PRO,
                [
                    IO.Combo.Input(
                        "aspect_ratio",
                        options=GEMINI_BASE_RATIOS,
                        default="auto",
                    ),
                    IO.Combo.Input(
                        "resolution",
                        options=["1K", "2K", "4K"],
                        default="1K",
                    ),
                ],
            ),
            IO.DynamicCombo.Option(GEMINI_NB2, [*nb2, *sampling()]),
            IO.DynamicCombo.Option(GEMINI_NB2_LITE, [*nb2_lite, *sampling()]),
        ],
    )


def _seedream_selector() -> Input:
    pro = _without(
        _seedream_model_inputs(
            max_ref_images=10,
            presets=RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
            max_width=3136,
            max_height=2496,
            supports_batch=False,
        ),
        "images",
    )
    lite = _without(
        _seedream_model_inputs(
            max_ref_images=14,
            presets=RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
            supports_batch=True,
        ),
        "images",
    )

    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option("seedream 5.0 pro", pro),
            IO.DynamicCombo.Option("seedream 5.0 lite", lite),
        ],
    )


def _flux2_selector() -> Input:
    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option(
                "Flux.2 [pro]",
                _without(_flux2_model_inputs(), "images"),
            ),
            IO.DynamicCombo.Option(
                "Flux.2 [max]",
                _without(_flux2_model_inputs(), "images"),
            ),
        ],
    )


# -----------------------------------------------------------------------------
# Settings node
# -----------------------------------------------------------------------------


class KASKIImageAPISettings(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAPISettings_KASKI",
            display_name="KASKI Image API Settings",
            category=CATEGORY,
            description=(
                "Shared settings for ComfyUI's built-in image API nodes. "
                "Provider execution remains inside comfy_api_nodes."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "backend",
                    options=[
                        IO.DynamicCombo.Option(
                            OPENAI_LABEL,
                            [
                                _openai_selector(),
                                IO.Int.Input(
                                    "n", default=1, min=1, max=8, step=1,
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            GEMINI_LABEL,
                            [
                                _gemini_selector(),
                                IO.Combo.Input(
                                    "response_modalities",
                                    options=["IMAGE", "IMAGE+TEXT"],
                                    default="IMAGE",
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            SEEDREAM_LABEL,
                            [
                                _seedream_selector(),
                                IO.Boolean.Input(
                                    "watermark", default=False, advanced=True,
                                ),
                                IO.Boolean.Input(
                                    "thinking", default=True, advanced=True,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            FLUX2_LABEL,
                            [_flux2_selector()],
                        ),
                    ],
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    advanced=True,
                ),
            ],
            outputs=[IO.Custom(SETTINGS_TYPE).Output(display_name="settings")],
        )

    @classmethod
    def execute(
        cls,
        backend: dict[str, Any],
        system_prompt: str,
    ) -> IO.NodeOutput:
        provider = backend["backend"]
        ms = backend["model_settings"]
        model_name = ms["model_settings"]

        if provider == OPENAI_LABEL:
            model = {
                "model": model_name,
                "size": ms["size"],
                "background": ms["background"],
                "quality": ms["quality"],
            }
            if model_name == "gpt-image-2":
                model |= {
                    "custom_width": int(ms["custom_width"]),
                    "custom_height": int(ms["custom_height"]),
                }
            return IO.NodeOutput({
                "provider": PROVIDER_OPENAI,
                "model": model,
                "n": int(backend["n"]),
            })

        if provider == GEMINI_LABEL:
            if model_name == GEMINI_PRO:
                model = {
                    "model": "gemini-3-pro-image-preview",
                    "aspect_ratio": ms["aspect_ratio"],
                    "resolution": ms["resolution"],
                }
                core = GEMINI_PRO
            else:
                model = {
                    "model": model_name,
                    "aspect_ratio": ms["aspect_ratio"],
                    "resolution": ms["resolution"],
                    "thinking_level": ms["thinking_level"],
                }
                core = GEMINI_NB2

            return IO.NodeOutput({
                "provider": PROVIDER_GEMINI,
                "core": core,
                "model": model,
                "response_modalities": backend["response_modalities"],
                "system_prompt": system_prompt,
                "temperature": float(ms.get("temperature", 1.0)),
                "top_p": float(ms.get("top_p", 0.95)),
            })

        if provider == SEEDREAM_LABEL:
            model = {
                "model": model_name,
                "size_preset": ms["size_preset"],
                "width": int(ms["width"]),
                "height": int(ms["height"]),
            }
            if model_name == "seedream 5.0 lite":
                model |= {
                    "max_images": int(ms.get("max_images", 1)),
                    "fail_on_partial": bool(ms.get("fail_on_partial", False)),
                }
            return IO.NodeOutput({
                "provider": PROVIDER_SEEDREAM,
                "model": model,
                "watermark": bool(backend["watermark"]),
                "thinking": bool(backend["thinking"]),
            })

        if provider == FLUX2_LABEL:
            return IO.NodeOutput({
                "provider": PROVIDER_FLUX2,
                "model": {
                    "model": model_name,
                    "width": int(ms["width"]),
                    "height": int(ms["height"]),
                },
            })

        raise ValueError(f"Unknown backend: {provider!r}")


# -----------------------------------------------------------------------------
# Generator node
# -----------------------------------------------------------------------------


class KASKIImageAPIGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAPIGenerator_KASKI",
            display_name="KASKI Image API Generator",
            category=CATEGORY,
            description=(
                "Thin router over ComfyUI's built-in OpenAI, Gemini, Seedream "
                "and FLUX.2 API nodes."
            ),
            inputs=[
                IO.String.Input("prompt", default="", multiline=True),
                IO.Custom(SETTINGS_TYPE).Input("settings"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                ),
                IO.Image.Input("images", optional=True),
                IO.Mask.Input("mask", optional=True),
                IO.Custom("GEMINI_INPUT_FILES").Input("files", optional=True),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.String.Output(display_name="text"),
                IO.Image.Output(display_name="thought_image"),
            ],
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
        settings: dict[str, Any],
        seed: int,
        images: Input.Image | None = None,
        mask: Input.Image | None = None,
        files: Any | None = None,
    ) -> IO.NodeOutput:
        try:
            if not isinstance(settings, dict):
                raise TypeError("settings must come from KASKI Image API Settings")

            provider = settings["provider"]
            image_group = _image_group(images)

            if provider == PROVIDER_OPENAI:
                model = dict(settings["model"])
                model["images"] = image_group
                model["mask"] = mask
                out = await _run_core(
                    OpenAIGPTImageNodeV2,
                    cls,
                    prompt=prompt,
                    model=model,
                    n=settings["n"],
                    seed=seed & 0x7FFFFFFF,
                )
                image = out[0]
                return IO.NodeOutput(image, "", _black_like(image))

            if provider == PROVIDER_GEMINI:
                if settings["core"] == GEMINI_PRO:
                    model = settings["model"]
                    out = await _run_core(
                        GeminiImage2,
                        cls,
                        prompt=prompt,
                        model=model["model"],
                        seed=seed,
                        aspect_ratio=model["aspect_ratio"],
                        resolution=model["resolution"],
                        response_modalities=settings["response_modalities"],
                        images=images,
                        files=files,
                        system_prompt=settings["system_prompt"],
                    )
                    image, text = out.args
                    return IO.NodeOutput(image, text, _black_like(image))

                model = dict(settings["model"])
                model["images"] = image_group
                model["files"] = files
                out = await _run_core(
                    GeminiNanoBanana2V2,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=seed,
                    response_modalities=settings["response_modalities"],
                    system_prompt=settings["system_prompt"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                )
                return IO.NodeOutput(*out.args)

            if provider == PROVIDER_SEEDREAM:
                model = dict(settings["model"])
                model["images"] = image_group
                out = await _run_core(
                    ByteDanceSeedreamNodeV2,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=seed & 0x7FFFFFFF,
                    watermark=settings["watermark"],
                    thinking=settings["thinking"],
                )
                image = out[0]
                return IO.NodeOutput(image, "", _black_like(image))

            if provider == PROVIDER_FLUX2:
                model = dict(settings["model"])
                model["images"] = image_group
                out = await _run_core(
                    Flux2ImageNode,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=seed,
                )
                image = out[0]
                return IO.NodeOutput(image, "", _black_like(image))

            raise ValueError(f"Unknown provider: {provider!r}")

        except Exception as error:
            _log_soft_error("KASKIImageAPIGenerator.execute", error)
            black = _black_image()
            return IO.NodeOutput(black, "", black)


UNIFIED_IMAGE_API_NODE_CLASS_MAPPINGS = {
    "ImageAPISettings_KASKI": KASKIImageAPISettings,
    "ImageAPIGenerator_KASKI": KASKIImageAPIGenerator,
}

UNIFIED_IMAGE_API_NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAPISettings_KASKI": "KASKI Image API Settings",
    "ImageAPIGenerator_KASKI": "KASKI Image API Generator",
}
