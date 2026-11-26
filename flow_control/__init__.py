from comfy_api.latest import ComfyExtension

from .selection import SELECTION_NODES_LIST
from .batches_lists import BATCHES_LISTS_NODES_LIST
from .API_related import API_RELATED_NODES_LIST


FLOW_CONTROL_NODES_LIST = SELECTION_NODES_LIST + BATCHES_LISTS_NODES_LIST + API_RELATED_NODES_LIST
