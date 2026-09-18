from comfy_api.latest import ComfyExtension
from .comfy_patches import apply_comfy_patches

from .async_tools import ASYNC_TOOLS_NODES_LIST
from .id_tools import ID_TOOLS_NODES_LIST
from .image_saver import IMAGE_SAVER_NODES_LIST
from .input_conform import INPUT_CONFORM_NODES_LIST
from .loaders import LOADERS_NODES_LIST
from .string_tools import STRING_TOOLS_NODES_LIST
from .api_adaptions import API_ADAPTIONS_NODES_LIST
from .video_saver import PRORES_SAVER_NODE_LIST
from .video_tools import VIDEO_TOOLS_NODES_LIST

#apply patches
apply_comfy_patches()

# V3-Inits
class KaskisComfyNodes(ComfyExtension):
    async def get_node_list(self):
        return [
            *ASYNC_TOOLS_NODES_LIST,
            *ID_TOOLS_NODES_LIST,
            *IMAGE_SAVER_NODES_LIST,
            *INPUT_CONFORM_NODES_LIST,
            *LOADERS_NODES_LIST,
            *STRING_TOOLS_NODES_LIST,
            *API_ADAPTIONS_NODES_LIST,
            *PRORES_SAVER_NODE_LIST,
            *VIDEO_TOOLS_NODES_LIST
        ]

async def comfy_entrypoint():
    return KaskisComfyNodes()