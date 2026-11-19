"""KASKI API adaptions."""

from .image_saver import SavePNGwithMetadata
from .image_tools import IMAGE_TOOLS_NODES_LIST

IMAGE_SAVER_NODES_LIST = [
    SavePNGwithMetadata,
]


IMAGE_NODES_LIST = IMAGE_SAVER_NODES_LIST + IMAGE_TOOLS_NODES_LIST