from comfy_api.latest import IO
import comfy_api_nodes.nodes_bytedance as bytedance

import logging


log = logging.getLogger(__name__)


def apply_comfy_patches():
    """
    Apply all KASKI runtime patches to ComfyUI.
    
    """

    _patch_api_retry_status()


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