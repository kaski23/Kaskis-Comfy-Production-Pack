"""KASKI API adaptions."""

from .json_to_dicts import JSON_TO_DICT_NODE_LIST
from .dicts_to_json import DICTS_TO_JSON_NODE_LIST
from .dicts_to_dicts import DICTS_TO_DICTS_NODE_LIST


JSON_TOOLS_NODE_LIST = (
        JSON_TO_DICT_NODE_LIST
        + DICTS_TO_JSON_NODE_LIST
        + DICTS_TO_DICTS_NODE_LIST)