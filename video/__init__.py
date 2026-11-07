"""KASKI API adaptions."""

from .video_saver import SaveProRes
from .video_tools import ExtendVideo, ShortenVideo, TemporalSmoother


VIDEO_SAVER_NODES_LIST = [
    SaveProRes,
]


VIDEO_TOOL_NODES_LIST = [
    ExtendVideo,
    ShortenVideo,
    TemporalSmoother,
]


VIDEO_NODES_LIST = VIDEO_SAVER_NODES_LIST + VIDEO_TOOL_NODES_LIST