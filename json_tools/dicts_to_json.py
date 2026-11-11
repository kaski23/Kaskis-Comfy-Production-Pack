import json
import re
from comfy_api.latest import io as IO

class GenerateJSONfromDICT(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateJSONfromDICT_KASKI",
            display_name="Generate JSON from DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                IO.Dict.Input(
                    "data",
                ),
                IO.Boolean.Input(
                    "pretty",
                    default=True,
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="JSON",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        data: dict,
        pretty: bool,
    ) -> IO.NodeOutput:

        if data is None:
            return IO.NodeOutput("")

        json_string = json.dumps(
            data,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )

        return IO.NodeOutput(json_string)


class GetValueFromDict(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GetValueFromDict_KASKI",
            display_name="Get Value from DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                IO.Dict.Input("data"),
                IO.String.Input(
                    "key",
                    default="",
                    multiline=False,
                    tooltip="Path, e.g. body_3.matching_hooks or body_3.matching_hooks[0].",
                ),
            ],
            outputs=[IO.String.Output(display_name="Values", is_output_list=True)],
        )

    @classmethod
    def execute(cls, data: dict, key: str) -> IO.NodeOutput:
        value = data
        not_found = f"No entry for DICT path {key!r}."

        for segment in key.strip().split("."):
            match = re.fullmatch(r"([^\[\]]+)((?:\[[0-9]+\])*)", segment)
            if match is None:
                raise ValueError(f"Invalid DICT path {key!r}.")

            field, brackets = match.groups()
            if not isinstance(value, dict) or field not in value:
                raise ValueError(not_found)
            value = value[field]

            for index in re.findall(r"\[([0-9]+)\]", brackets):
                if not isinstance(value, list):
                    raise ValueError(not_found)
                try:
                    value = value[int(index)]
                except (IndexError, ValueError):
                    raise ValueError(not_found) from None

        items = value if isinstance(value, list) else [value]
        strings = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in items
        ]
        return IO.NodeOutput(strings)

DICTS_TO_JSON_NODE_LIST = [
    GenerateJSONfromDICT,
    GetValueFromDict,
]