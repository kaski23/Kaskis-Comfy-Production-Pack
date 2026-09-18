"""KASKI API adaptions."""

from .videoapi_text_to_video import KASKITextToVideoAPI
from .videoapi_first_last_to_video import KASKIFirstLastFrameToVideoAPI
from .videoapi_reference_to_video import KASKIReferenceToVideoAPI
from .videoapi_enhance import KASKIVideoEnhanceAPI
from .imageapi_settings_generator import KASKIImageAPISettings, KASKIImageAPIGenerator

IMAGEAPI_NODES_LIST = [
    KASKIImageAPISettings,
    KASKIImageAPIGenerator,
]


VIDEOAPI_NODES_LIST = [
    KASKITextToVideoAPI,
    KASKIFirstLastFrameToVideoAPI,
    KASKIReferenceToVideoAPI,
    KASKIVideoEnhanceAPI,
]


API_ADAPTIONS_NODES_LIST = IMAGEAPI_NODES_LIST + VIDEOAPI_NODES_LIST