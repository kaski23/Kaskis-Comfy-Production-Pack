"""KASKI unified image API adapter.

Delegates generation, uploads, validation and response decoding to ComfyUI's
built-in API nodes. KASKI owns only the shared settings, routing and outputs.

Output contract (unchanged node IDs and socket positions):
    image, thoughts, thought_image, prompt, modelName, seed

The last three outputs are strings suitable for the KASKI PNG saver.
The seed is the value passed to the selected Comfy core node, not a promise
that the remote service implements deterministic generation.

Provider interfaces checked against Comfy-Org/ComfyUI master, 2026-09-16.
Private upstream widget helpers are intentionally reused rather than forked.
If a helper or execute signature changes, update this adapter, not the API
implementation. This module makes no HTTP requests of its own.
"""

from __future__ import annotations

import inspect
import traceback
from typing import Any

import torch

from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.bytedance import (
    RECOMMENDED_PRESETS_SEEDREAM_5_LITE,
    RECOMMENDED_PRESETS_SEEDREAM_5_PRO,
)
from comfy_api_nodes.nodes_bfl import Flux2ImageNode, _flux2_model_inputs
from comfy_api_nodes.nodes_bytedance import (
    SEEDREAM_MODELS,
    ByteDanceSeedreamNodeV3,
    _seedream_model_inputs,
)
from comfy_api_nodes.nodes_gemini import (
    GEMINI_IMAGE_SYS_PROMPT,
    GeminiImage2,
    GeminiNanoBanana2V2,
    _nano_banana_2_v2_model_inputs,
)
from comfy_api_nodes.nodes_openai import (
    GPT_IMAGE_25_QUALITIES,
    GPT_IMAGE_QUALITIES,
    OpenAIGPTImageNodeV2,
    _gpt_image_2_model_inputs,
    _gpt_image_legacy_model_inputs,
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

# These labels are translated by the current Gemini core implementations.
# Keep the same names and IDs here only to record accurate provenance.
GEMINI_MODEL_IDS = {
    "gemini-3-pro-image-preview": "gemini-3-pro-image",
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image",
    "Nano Banana 2 Lite": "gemini-3.1-flash-lite-image",
}

# The Flux.2 core routes these two labels to /flux-2-pro/generate and
# /flux-2-max/generate. These are metadata identifiers, not request logic.
FLUX2_MODEL_IDS = {
    "Flux.2 [pro]": "flux-2-pro",
    "Flux.2 [max]": "flux-2-max",
}

# -----------------------------------------------------------------------------
# Generic glue / output normalization
# -----------------------------------------------------------------------------


def _black_image(width: int = 1024, height: int = 1024) -> torch.Tensor:
    return torch.zeros((1, height, width, 3), dtype=torch.float32)


def _black_like(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
        return _black_image()
    h, w = (image.shape[1], image.shape[2]) if image.ndim == 4 else image.shape[:2]
    dtype = image.dtype if image.is_floating_point() else torch.float32
    return torch.zeros((1, int(h), int(w), 3), dtype=dtype, device=image.device)


def _log_soft_error(where: str, error: Exception) -> None:
    print(f"[KASKI Image API] {where}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _image_group(images: Input.Image | None) -> dict[str, Input.Image]:
    # A single socket may contain an entire BHWC batch. Upstream flattens it.
    return {} if images is None else {"image_1": images}


def _without(inputs: list[Input], *ids: str) -> list[Input]:
    """Reuse Comfy widgets, excluding sockets owned by KASKI Generator."""
    blocked = set(ids)
    return [item for item in inputs if getattr(item, "id", None) not in blocked]


async def _run_core(
    core_node: type[IO.ComfyNode],
    active_cls: type[IO.ComfyNode],
    **kwargs: Any,
) -> IO.NodeOutput:
    """Run the original core classmethod with KASKI's active auth context.

    The executing node is KASKI, not the imported provider class. Passing the
    active class lets upstream sync_op/upload helpers find the hidden auth,
    API-key and unique-id context attached by Comfy's execution machinery.
    """
    func = getattr(core_node.execute, "__func__", None)
    if func is None:
        raise RuntimeError(f"{core_node.__name__}.execute is no longer a classmethod.")
    result = func(active_cls, **kwargs)
    return await result if inspect.isawaitable(result) else result


def _core_values(result: Any, expected: int, provider: str) -> tuple[Any, ...]:
    """Unwrap a core result without accidentally indexing an IMAGE tensor."""
    values = result.args if isinstance(result, IO.NodeOutput) else result
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{provider}: unsupported core output type {type(result).__name__}")
    if len(values) != expected:
        raise RuntimeError(
            f"{provider}: expected {expected} core outputs, received {len(values)}. "
            "The upstream node's output contract may have changed."
        )
    return tuple(values)


def _normalized_output(
    image: torch.Tensor,
    thoughts: str | None,
    thought_image: torch.Tensor | None,
    prompt: str,
    model_name: str,
    seed: int | str,
) -> IO.NodeOutput:
    """The one and only six-output boundary for every provider."""
    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise TypeError("Core image output must be a BHWC torch.Tensor.")
    if image.shape[0] < 1:
        raise ValueError("Core returned an empty image batch.")
    if thought_image is None:
        thought_image = _black_like(image)
    if not isinstance(thought_image, torch.Tensor) or thought_image.ndim != 4:
        raise TypeError("Core thought_image must be a BHWC torch.Tensor.")
    if seed is None or seed == "":
        seed = -1
        
    return IO.NodeOutput(
        image,
        "" if thoughts is None else str(thoughts),
        thought_image,
        str(prompt),
        str(model_name),
        int(seed),
    )


def _model_id(provider: str, selected: str) -> str:
    """Resolve the selected model to the identifier used by the core."""
    if provider == PROVIDER_GEMINI:
        return GEMINI_MODEL_IDS.get(selected, selected)
    if provider == PROVIDER_SEEDREAM:
        return SEEDREAM_MODELS[selected]
    if provider == PROVIDER_FLUX2:
        return FLUX2_MODEL_IDS.get(selected, selected)
    return selected



def _openai_selector() -> Input:
    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option(
                "gpt-image-2.5-flare",
                _without(
                    _gpt_image_2_model_inputs(
                        ("auto", "opaque", "transparent"),
                        GPT_IMAGE_25_QUALITIES,
                    ),
                    "images",
                    "mask",
                ),
            ),
            IO.DynamicCombo.Option(
                "gpt-image-2.5-sunburst",
                _without(
                    _gpt_image_2_model_inputs(
                        ("auto", "opaque", "transparent"),
                        GPT_IMAGE_25_QUALITIES,
                    ),
                    "images",
                    "mask",
                ),
            ),
            IO.DynamicCombo.Option(
                "gpt-image-2",
                _without(
                    _gpt_image_2_model_inputs(
                        ("auto", "opaque"),
                        GPT_IMAGE_QUALITIES,
                    ),
                    "images",
                    "mask",
                ),
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
                        default="2K",
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
            supports_fast=True,
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
            display_name="KASKI Unified Image API Settings",
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
            if "custom_width" in ms and "custom_height" in ms:
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
            if model_name == "seedream 5.0 pro":
                model["prompt_optimization"] = ms.get("prompt_optimization", "standard")
            elif model_name == "seedream 5.0 lite":
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
            display_name="KASKI Unified Image API",
            category=CATEGORY,
            description=(
                "Thin router over ComfyUI's built-in OpenAI, Gemini, Seedream "
                "and FLUX.2 API nodes. Outputs image and generation metadata."
            ),
            inputs=[
                IO.String.Input("prompt", default="", multiline=True),
                IO.Custom(SETTINGS_TYPE).Input("settings"),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
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
                IO.String.Output(display_name="thoughts"),
                IO.Image.Output(display_name="thought_image"),
                IO.String.Output(display_name="prompt"),
                IO.String.Output(display_name="modelName"),
                IO.Int.Output(display_name="seed"),
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
            selected_model = settings["model"]["model"]
            model_name = _model_id(provider, selected_model)
            effective_seed = int(seed) % (2**31)

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
                    seed=effective_seed,
                )
                (image,) = _core_values(out, 1, provider)
                return _normalized_output(
                    image, "", None, prompt, model_name, effective_seed
                )

            if provider == PROVIDER_GEMINI:
                if settings["core"] == GEMINI_PRO:
                    model = settings["model"]
                    out = await _run_core(
                        GeminiImage2,
                        cls,
                        prompt=prompt,
                        model=model["model"],
                        seed=effective_seed,
                        aspect_ratio=model["aspect_ratio"],
                        resolution=model["resolution"],
                        response_modalities=settings["response_modalities"],
                        images=images,
                        files=files,
                        system_prompt=settings["system_prompt"],
                    )
                    image, text = _core_values(out, 2, provider)
                    return _normalized_output(
                        image, text, None, prompt, model_name, effective_seed
                    )

                model = dict(settings["model"])
                model["images"] = image_group
                model["files"] = files
                out = await _run_core(
                    GeminiNanoBanana2V2,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=effective_seed,
                    response_modalities=settings["response_modalities"],
                    system_prompt=settings["system_prompt"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                )
                image, text, thought_image = _core_values(out, 3, provider)
                return _normalized_output(
                    image, text, thought_image, prompt, model_name, effective_seed
                )

            if provider == PROVIDER_SEEDREAM:
                model = dict(settings["model"])
                model["images"] = image_group
                out = await _run_core(
                    ByteDanceSeedreamNodeV3,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=effective_seed,
                    watermark=settings["watermark"],
                    thinking=settings["thinking"],
                )
                (image,) = _core_values(out, 1, provider)
                return _normalized_output(
                    image, "", None, prompt, model_name, effective_seed
                )

            if provider == PROVIDER_FLUX2:
                model = dict(settings["model"])
                model["images"] = image_group
                out = await _run_core(
                    Flux2ImageNode,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=effective_seed,
                )
                (image,) = _core_values(out, 1, provider)
                return _normalized_output(
                    image, "", None, prompt, model_name, effective_seed
                )

            raise ValueError(f"Unknown provider: {provider!r}")

        except Exception as error:
            # Preserve the original KASKI soft-error behavior. An error is
            # explicitly marked in thoughts, and provenance is left empty.
            # The returned black image is a placeholder, not an API result.
            _log_soft_error("KASKIImageAPIGenerator.execute", error)
            black = _black_image()
            return _normalized_output(
                black,
                f"KASKI API error: {type(error).__name__}: {error}",
                black,
                "",
                "",
                "",
            )