"""KASKI unified image API adapter.

Delegates generation, uploads, validation and response decoding to ComfyUI's
built-in API nodes. KASKI owns only the shared settings, routing and outputs.

Output contract (unchanged node IDs and socket positions):
    image, thoughts, thought_image, prompt, modelName, seed

The prompt and modelName outputs are strings; seed is an INT.
The seed is the value passed to the selected Comfy core node, not a promise
that the remote service implements deterministic generation.

Provider interfaces checked against Comfy-Org/ComfyUI master, 2026-10-11.
Upstream commit: 7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217.
Model lists, widgets and defaults are read from the installed core schemas.
KASKI copies schema objects before adapting socket ownership. New model settings
are passed through unchanged; execute signature changes still require an update. This module makes no HTTP requests of its own.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
import traceback
from typing import Any

import torch

from comfy_api.latest import IO, Input
from comfy_api_nodes.nodes_bfl import Flux2ImageNode
from comfy_api_nodes.nodes_bytedance import SEEDREAM_MODELS, ByteDanceSeedreamNodeV3
from comfy_api_nodes.nodes_gemini import GeminiImage2, GeminiNanoBanana2V2
from comfy_api_nodes.nodes_openai import OpenAIGPTImageNodeV2



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
GEMINI_NB21 = "Gemini Nano Banana 2.1"
GEMINI_NB2 = "Nano Banana 2 (Gemini 3.1 Flash Image)"
GEMINI_NB2_LITE = "Nano Banana 2 Lite"

# These labels are translated by the current Gemini core implementations.
# Keep the same names and IDs here only to record accurate provenance.
GEMINI_MODEL_IDS = {
    GEMINI_NB21: "gemini-nano-banana-2.1",
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


def _core_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Separate routing metadata from upstream execution arguments."""
    return {key: value for key, value in settings.items()
            if key not in {"provider", "core", "model"}}


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



# These sockets belong to the KASKI generator, never to provider settings.
_GENERATOR_INPUTS = {"prompt", "seed", "images", "mask", "files"}


def _core_input(core_node: type[IO.ComfyNode], input_id: str) -> Input:
    """Copy an upstream widget; never mutate the provider's own schema."""
    for widget in core_node.define_schema().inputs:
        if widget.id == input_id:
            return deepcopy(widget)
    raise RuntimeError(f"{core_node.__name__} no longer defines input {input_id!r}.")


def _settings_inputs(inputs: list[Input]) -> list[Input]:
    """Strip generator sockets recursively, retaining upstream widget details."""
    result = []
    for widget in deepcopy(inputs):
        if widget.id in _GENERATOR_INPUTS:
            continue
        if isinstance(widget, IO.DynamicCombo.Input):
            for option in widget.options:
                option.inputs = _settings_inputs(option.inputs)
        result.append(widget)
    return result


def _model_selector(core_node: type[IO.ComfyNode]) -> Input:
    selector = _core_input(core_node, "model")
    if not isinstance(selector, IO.DynamicCombo.Input):
        raise RuntimeError(f"{core_node.__name__}.model is no longer a DynamicCombo.")
    selector.id = "model_settings"
    return _settings_inputs([selector])[0]


def _provider_extras(core_node: type[IO.ComfyNode]) -> list[Input]:
    """Provider-level widgets not already owned by the shared KASKI UI."""
    return _settings_inputs(_without(
        core_node.define_schema().inputs, "model", "system_prompt"
    ))


def _gemini_selector() -> Input:
    selector = _model_selector(GeminiNanoBanana2V2)
    # Sampling and any future provider-level settings belong to the NB branch.
    extras = _without(_provider_extras(GeminiNanoBanana2V2), "response_modalities")
    for option in selector.options:
        option.inputs.extend(deepcopy(extras))
    pro = _without(_provider_extras(GeminiImage2), "response_modalities")
    selector.options.insert(0, IO.DynamicCombo.Option(GEMINI_PRO, pro))
    return selector


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
                                _model_selector(OpenAIGPTImageNodeV2),
                                *_provider_extras(OpenAIGPTImageNodeV2),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            GEMINI_LABEL,
                            [
                                _gemini_selector(),
                                _core_input(GeminiNanoBanana2V2, "response_modalities"),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            SEEDREAM_LABEL,
                            [
                                _model_selector(ByteDanceSeedreamNodeV3),
                                *_provider_extras(ByteDanceSeedreamNodeV3),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            FLUX2_LABEL,
                            [
                                _model_selector(Flux2ImageNode),
                                *_provider_extras(Flux2ImageNode),
                            ],
                        ),
                    ],
                ),
                _core_input(GeminiNanoBanana2V2, "system_prompt"),
            ],
            outputs=[IO.Custom(SETTINGS_TYPE).Output(display_name="settings")],
        )

    @classmethod
    def execute(
        cls,
        backend: dict[str, Any],
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        provider = backend["backend"]
        ms = backend["model_settings"]
        model_name = ms["model_settings"]

        # Keep every current/future model widget rather than enumerating fields.
        model = {"model": model_name, **{
            key: value for key, value in ms.items() if key != "model_settings"
        }}
        common = {
            key: value for key, value in backend.items()
            if key not in {"backend", "model_settings"}
        }

        if provider == OPENAI_LABEL:
            return IO.NodeOutput({
                **common, "provider": PROVIDER_OPENAI, "model": model,
            })

        if provider == GEMINI_LABEL:
            core = GEMINI_PRO if model_name == GEMINI_PRO else GEMINI_NB2
            if core == GEMINI_PRO:
                model["model"] = "gemini-3-pro-image-preview"
            return IO.NodeOutput({
                **common,
                "provider": PROVIDER_GEMINI,
                "core": core,
                "model": model,
                "system_prompt": system_prompt,
            })

        if provider == SEEDREAM_LABEL:
            return IO.NodeOutput({
                **common, "provider": PROVIDER_SEEDREAM, "model": model,
            })

        if provider == FLUX2_LABEL:
            return IO.NodeOutput({
                **common, "provider": PROVIDER_FLUX2, "model": model,
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
                    seed=effective_seed,
                    **_core_settings(settings),
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
                        seed=effective_seed,
                        images=images,
                        files=files,
                        **model,
                        **_core_settings(settings),
                    )
                    image, text = _core_values(out, 2, provider)
                    return _normalized_output(
                        image, text, None, prompt, model_name, effective_seed
                    )

                model = dict(settings["model"])
                model["images"] = image_group
                model["files"] = files
                # Sampling widgets are top-level arguments in the core API,
                # but nested under each model in the KASKI settings UI.
                extra_ids = {widget.id for widget in
                             _provider_extras(GeminiNanoBanana2V2)}
                extras = {key: model.pop(key) for key in list(model)
                          if key in extra_ids}
                out = await _run_core(
                    GeminiNanoBanana2V2,
                    cls,
                    prompt=prompt,
                    model=model,
                    seed=effective_seed,
                    **{**_core_settings(settings), **extras},
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
                    **_core_settings(settings),
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
                    **_core_settings(settings),
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