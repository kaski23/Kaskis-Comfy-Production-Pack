"""KASKI API adaptions."""

from .videoapi_text_to_video import KASKITextToVideoAPI
from .videoapi_first_last_to_video import KASKIFirstLastFrameToVideoAPI
from .videoapi_reference_to_video import KASKIReferenceToVideoAPI
from .videoapi_enhance import KASKIVideoEnhanceAPI
from .imageapi_settings_generator import IMAGEAPI_NODES_LIST
from .videoapi_seedance_draft import SEEDANCE_DRAFT_API_NODES_LIST
from .elevenlabsapi_dialogue import KASKIElevenLabsJSONToDialogue


SINGLE_CLASSES_NODE_LIST = [
    KASKITextToVideoAPI,
    KASKIFirstLastFrameToVideoAPI,
    KASKIReferenceToVideoAPI,
    KASKIVideoEnhanceAPI,
    KASKIElevenLabsJSONToDialogue,
]


API_ADAPTIONS_NODES_LIST = IMAGEAPI_NODES_LIST + SINGLE_CLASSES_NODE_LIST + SEEDANCE_DRAFT_API_NODES_LIST