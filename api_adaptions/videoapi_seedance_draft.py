"""KASKI wrappers for the native ComfyUI Seedance 2.5 Draft and Final nodes."""

import asyncio
import logging
from copy import copy
from dataclasses import replace
from fractions import Fraction

import torch

from comfy_api.latest import ComfyExtension, IO, InputImpl, Types
from comfy_api_nodes.nodes_bytedance import (
    ByteDance2DraftToFinalVideoNode,
    ByteDance2ReferenceNodeV2,
)
from comfy_api_nodes.util._helpers import sleep_with_interrupt
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted

from .videoapi_common import CATEGORY, _black_video


log = logging.getLogger(__name__)

DRAFT_MODEL = "Seedance 2.5 Draft"
FINAL_MODEL = "Seedance 2.5"
SUCCESS = "NO ERRORS TODAY - SUCCEEDED"
MAX_DRAFT_RUNS = 32
DEFAULT_RUN_DELAY_MS = 100
MAX_RUN_DELAY_MS = 60000
SEED_MODULUS = 2**31



def _draft_price_badge(native_badge):
    if native_badge is None:
        return None

    return replace(
        native_badge,
        depends_on=replace(
            native_badge.depends_on,
            widgets=[*native_badge.depends_on.widgets, "n_runs"],
        ),
        expr=f"""
        (
          $cost := {native_badge.expr};
          $runs := $exists(widgets.n_runs) ? widgets.n_runs : 1;
          $cost.type = "usd"
            ? $merge([$cost, {{"usd": $cost.usd * $runs}}])
            : $cost.type = "range_usd"
              ? $merge([$cost, {{"min_usd": $cost.min_usd * $runs, "max_usd": $cost.max_usd * $runs}}])
              : $cost
        )
        """,
    )


def _fixed_reference_inputs(inputs):
    for input_spec in inputs:
        if not isinstance(input_spec, IO.Autogrow.Input):
            yield input_spec
            continue

        for index, template_input in enumerate(input_spec.template.get_all()):
            reference = copy(template_input)
            # Preserve the socket path used by saved workflows and the native model dict.
            reference.id = f"{input_spec.id}.{template_input.id}"
            reference.optional = template_input.optional or index >= input_spec.template.min
            yield reference


class KASKISeedance25Draft(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        native = ByteDance2ReferenceNodeV2.define_schema()
        selector = copy(next(item for item in native.inputs if item.id == "model"))
        draft_option = copy(next(option for option in selector.options if option.key == DRAFT_MODEL))
        draft_option.inputs = list(_fixed_reference_inputs(draft_option.inputs))
        selector.options = [draft_option]
        selector.tooltip = "Seedance 2.5 Draft renders a 480p preview for Draft to Final."

        outputs = [
            *native.outputs,
            IO.String.Output(display_name="prompt"),
            IO.String.Output(display_name="modelName"),
            IO.Int.Output(display_name="seed"),
            IO.String.Output(display_name="error"),
        ]
        for output in outputs:
            output.is_output_list = True

        return replace(
            native,
            node_id="Seedance25DraftAPI_KASKI",
            display_name="KASKI Seedance 2.5 Draft",
            category=CATEGORY,
            description="Generate parallel Seedance 2.5 drafts and return one list item per run.",
            inputs=[
                *(selector if item.id == "model" else item for item in native.inputs),
                IO.Int.Input(
                    "n_runs",
                    default=1,
                    min=1,
                    max=MAX_DRAFT_RUNS,
                    step=1,
                    optional=True,
                    tooltip="Number of parallel drafts. Each run increments the seed by one.",
                ),
                IO.Int.Input(
                    "run_delay_ms",
                    default=DEFAULT_RUN_DELAY_MS,
                    min=0,
                    max=MAX_RUN_DELAY_MS,
                    step=1,
                    optional=True,
                    tooltip="Milliseconds between draft starts. The first starts immediately. "
                    "Use 0 to start all runs together.",
                ),
            ],
            outputs=outputs,
            price_badge=_draft_price_badge(native.price_badge),
        )

    @classmethod
    async def execute(
        cls,
        model: dict,
        seed: int,
        watermark: bool,
        n_runs: int = 1,
        run_delay_ms: int = DEFAULT_RUN_DELAY_MS,
    ) -> IO.NodeOutput:
        for name, value, minimum, maximum in (
            ("n_runs", n_runs, 1, MAX_DRAFT_RUNS),
            ("run_delay_ms", run_delay_ms, 0, MAX_RUN_DELAY_MS),
        ):
            if not isinstance(value, int) or not minimum <= value <= maximum:
                message = f"ValueError: {name} must be between {minimum} and {maximum}."
                return IO.NodeOutput(
                    [_black_video()], [""], [model.get("prompt", "")],
                    [DRAFT_MODEL], [seed], [message],
                )

        tasks = [
            asyncio.create_task(
                cls._run_draft(
                    model, (seed + index) % SEED_MODULUS, watermark,
                    start_delay=index * run_delay_ms / 1000,
                )
            )
            for index in range(n_runs)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            # Stop and drain sibling tasks on interruption or cancellation.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        return IO.NodeOutput(*(list(column) for column in zip(*results)))

    @classmethod
    async def _run_draft(cls, model: dict, seed: int, watermark: bool, *, start_delay: float):
        prompt = ""
        try:
            draft = dict(model, model=DRAFT_MODEL)
            prompt = draft.get("prompt", "")
            if start_delay:
                await sleep_with_interrupt(start_delay, cls)

            # Each run needs its own mapping of connected references.
            for key in ("reference_images", "reference_videos", "reference_audios", "reference_assets"):
                draft[key] = {
                    name: value for name, value in (draft.get(key) or {}).items() if value is not None
                }

            # Comfy's API client keys estimates and credit tracking by class.
            # Give each run its own key and retain this node's auth and ID.
            context = type("_SeedanceDraftRun", (ByteDance2ReferenceNodeV2,), {"hidden": cls.hidden})
            result = await ByteDance2ReferenceNodeV2.execute.__func__(
                context, model=draft, seed=seed, watermark=watermark
            )
            video, draft_task_id = result.result
            return video, draft_task_id, prompt, DRAFT_MODEL, seed, SUCCESS
        except ProcessingInterrupted:
            raise
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            log.warning("[KASKI Seedance 2.5 Draft seed=%s] %s", seed, message)
            return _black_video(), "", prompt, DRAFT_MODEL, seed, message


class KASKISeedance25DraftToFinal(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        native = ByteDance2DraftToFinalVideoNode.define_schema()
        return replace(
            native,
            node_id="Seedance25DraftToFinalAPI_KASKI",
            display_name="KASKI Seedance 2.5 Draft to Final Video",
            category=CATEGORY,
            outputs=[
                *native.outputs,
                IO.String.Output(display_name="draft_task_id"),
                IO.String.Output(display_name="modelName"),
                IO.String.Output(display_name="error"),
            ],
        )

    @classmethod
    async def execute(
        cls,
        draft_task_id: str,
        watermark: bool,
        resolution: str = "1080p",
    ) -> IO.NodeOutput:
        task_id = ""
        try:
            task_id = (draft_task_id or "").strip()
            if not task_id:
                raise ValueError("draft_task_id is empty.")

            result = await ByteDance2DraftToFinalVideoNode.execute.__func__(
                cls, draft_task_id=task_id, watermark=watermark, resolution=resolution
            )
            return IO.NodeOutput(result[0], task_id, FINAL_MODEL, SUCCESS)
        except ProcessingInterrupted:
            raise
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            log.warning("[KASKI Seedance 2.5 Draft to Final] %s", message)
            return IO.NodeOutput(_black_video(), task_id, FINAL_MODEL, message)


SEEDANCE_DRAFT_API_NODES_LIST = [
    KASKISeedance25Draft,
    KASKISeedance25DraftToFinal,
]
