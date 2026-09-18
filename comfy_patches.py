from comfy_api.latest import IO
import comfy_api_nodes.nodes_bytedance as bytedance

import logging


log = logging.getLogger(__name__)


def apply_comfy_patches():
    """
    Apply all KASKI runtime patches to ComfyUI.
    
    """

    _patch_api_retry_status()
    _patch_seedance25_mov()


def _patch_api_retry_status():
    """
    Add HTTP 402 to ComfyUI's retryable API status codes.

    """

    try:
        from comfy_api_nodes.util import client

        if not hasattr(client, "_RETRY_STATUS"):
            log.warning(
                "[KASKI] Retry patch skipped: "
                "comfy_api_nodes.util.client._RETRY_STATUS was not found."
            )
            return

        client._RETRY_STATUS.add(402)

        log.info(
            "[KASKI] Patched Comfy API retry statuses: %s",
            client._RETRY_STATUS,
        )

    except Exception:
        log.exception(
            "[KASKI] Failed to patch Comfy API retry statuses."
        )


def _patch_seedance25_mov():
    """
    Add MOV as an output option for Seedance 2.5.

    The original ComfyUI helper still creates the full Seedance input schema.
    KASKI only replaces its output_format input afterwards.

    This keeps future upstream changes to the rest of the schema intact.
    """

    try:
        if not hasattr(bytedance, "_seedance25_text_inputs"):
            log.warning(
                "[KASKI] Seedance MOV patch skipped: "
                "_seedance25_text_inputs was not found."
            )
            return

        current = bytedance._seedance25_text_inputs

        # Avoid wrapping the helper multiple times during reloads.
        if getattr(
            current,
            "_kaski_mov_patch",
            False,
        ):
            return

        original = current

        def patched(*args, **kwargs):
            inputs = original(
                *args,
                **kwargs,
            )

            found = False

            for index, input_def in enumerate(inputs):
                if input_def.id != "output_format":
                    continue

                inputs[index] = IO.Combo.Input(
                    "output_format",
                    options=[
                        "mp4",
                        "mov",
                    ],
                    default="mov",
                    tooltip="Container format of the output video.",
                )

                found = True
                break

            if not found:
                log.warning(
                    "[KASKI] Seedance MOV patch could not find "
                    "the output_format input."
                )

            return inputs

        patched._kaski_mov_patch = True
        patched._kaski_original = original

        bytedance._seedance25_text_inputs = patched

        log.info(
            "[KASKI] Patched Seedance 2.5 output formats: mp4, mov."
        )

    except Exception:
        log.exception(
            "[KASKI] Failed to patch Seedance 2.5 MOV support."
        )