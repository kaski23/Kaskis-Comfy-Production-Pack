"""KASKI API adaptions."""

from .image_saver import SavePNGwithMetadata
from .image_tools import ImagesInpaintOutpaint, CombineImagesToImageList

IMAGE_SAVER_NODES_LIST = [
    SavePNGwithMetadata,
]


IMAGE_TOOL_NODES_LIST = [
    ImagesInpaintOutpaint,
    CombineImagesToImageList,
]


IMAGE_NODES_LIST = IMAGE_SAVER_NODES_LIST + IMAGE_TOOL_NODES_LIST