"""Provider-agnostic helpers for KASKI video API nodes."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable

import torch

from comfy_api.latest import IO, Input, InputImpl, Types


CATEGORY = "KASKI/api-adaptions/video"
FALLBACK_PROVIDER_TOOLTIP = "Provider-specific setting inherited from ComfyUI."


@dataclass(frozen=True)
class Variant:
    """One selectable provider/model configuration.

    selector_id / selector_value describe the original ComfyUI model selector.
    Dynamic selectors rebuild nested dictionaries; ordinary selectors stay flat.
    """

    label: str
    node_cls: type[IO.ComfyNode]

    selector_id: str
    selector_value: str
    selector_is_dynamic: bool

    nested_ids: frozenset[str]
    inputs: tuple[Input, ...]

    route: str


def _schema_input(
    node_cls: type[IO.ComfyNode],
    input_id: str,
) -> Input:
    """Find one input definition inside an original ComfyUI node schema."""
    for item in node_cls.define_schema().inputs:
        if item.id == input_id:
            return item

    raise RuntimeError(
        f"{node_cls.__name__} no longer exposes input {input_id!r}."
    )


def _copy_input(item: Input) -> Input:
    """Clone a Comfy input and attach a fallback tooltip if needed."""
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
    """Expand a Comfy DynamicCombo selector into KASKI model variants."""
    schema = node_cls.define_schema()
    selector = _schema_input(node_cls, selector_id)

    if not isinstance(selector, IO.DynamicCombo.Input):
        raise TypeError(
            f"{node_cls.__name__}.{selector_id} is no longer a DynamicCombo."
        )

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
    """Expand a normal Comfy Combo selector into KASKI model variants."""
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


def _model_selector(
    variants: tuple[Variant, ...],
    tooltip: str,
) -> Input:
    """Build a model selector from prepared variants."""
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
            for variant in variants
        ],
        tooltip=tooltip,
    )


def _build_core_params(
    variant: Variant,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild the argument structure expected by the original provider node."""
    settings = dict(settings)

    if variant.selector_is_dynamic:
        nested_model = {
            variant.selector_id: variant.selector_value,
        }

        for key in variant.nested_ids:
            if key in settings:
                nested_model[key] = settings.pop(key)

        params = {
            variant.selector_id: nested_model,
        }
        params.update(settings)
        return params

    params = {
        variant.selector_id: variant.selector_value,
    }
    params.update(settings)
    return params


async def _run_core(
    node_cls: type[IO.ComfyNode],
    active_cls: type[IO.ComfyNode],
    **params: Any,
):
    """Run the original ComfyUI execute() function with KASKI as cls."""
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
    """Return output 0 from a wrapped ComfyUI node."""
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
    """Return a one-second black VIDEO for soft-error fallback."""
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


def _hidden_inputs():
    """Hidden ComfyUI API auth/runtime context required by wrapped nodes."""
    return [
        IO.Hidden.auth_token_comfy_org,
        IO.Hidden.api_key_comfy_org,
        IO.Hidden.unique_id,
    ]
