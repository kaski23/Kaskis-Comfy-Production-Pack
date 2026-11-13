"""KASKI wrappers for the native ComfyUI Seedance 2.5 Draft and Final nodes."""

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
from comfy_api_nodes.util.common_exceptions import ProcessingInterrupted

from .videoapi_common import CATEGORY, _black_video


log = logging.getLogger(__name__)

DRAFT_MODEL = "Seedance 2.5 Draft"
FINAL_MODEL = "Seedance 2.5"
SUCCESS = "NO ERRORS TODAY - SUCCEEDED"



class KASKISeedance25Draft(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        native = ByteDance2ReferenceNodeV2.define_schema()
        selector = copy(next(item for item in native.inputs if item.id == "model"))
        selector.options = [next(option for option in selector.options if option.key == DRAFT_MODEL)]
        selector.tooltip = "Seedance 2.5 Draft renders a 480p preview for Draft to Final."

        return replace(
            native,
            node_id="Seedance25DraftAPI_KASKI",
            display_name="KASKI Seedance 2.5 Draft",
            category=CATEGORY,
            description="Generate a Seedance 2.5 reference-to-video draft and return its task ID.",
            inputs=[selector if item.id == "model" else item for item in native.inputs],
            outputs=[
                *native.outputs,
                IO.String.Output(display_name="prompt"),
                IO.String.Output(display_name="modelName"),
                IO.Int.Output(display_name="seed"),
                IO.String.Output(display_name="error"),
            ],
        )

    @classmethod
    async def execute(cls, model: dict, seed: int, watermark: bool) -> IO.NodeOutput:
        prompt = ""
        try:
            draft = dict(model, model=DRAFT_MODEL)
            prompt = draft.get("prompt", "")
            # The native node replaces prepared refs in these mappings in place.
            for key in ("reference_images", "reference_videos"):
                if key in draft:
                    draft[key] = dict(draft[key])

            # Use this execution's hidden auth and node ID in the native method.
            result = await ByteDance2ReferenceNodeV2.execute.__func__(
                cls, model=draft, seed=seed, watermark=watermark
            )
            video, draft_task_id = result.result
            return IO.NodeOutput(video, draft_task_id, prompt, DRAFT_MODEL, seed, SUCCESS)
        except ProcessingInterrupted:
            raise
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            log.warning("[KASKI Seedance 2.5 Draft] %s", message)
            return IO.NodeOutput(_black_video(), "", prompt, DRAFT_MODEL, seed, message)


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
