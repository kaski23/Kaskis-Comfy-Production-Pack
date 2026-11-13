"""KASKI API adaptions."""

from .video_saver import VIDEO_SAVER_NODES_LIST
from .video_tools import VIDEO_TOOL_NODES_LIST
from .video_loader import VIDEO_LOADER_NODES_LIST








VIDEO_NODES_LIST = VIDEO_SAVER_NODES_LIST + VIDEO_TOOL_NODES_LIST + VIDEO_LOADER_NODES_LIST