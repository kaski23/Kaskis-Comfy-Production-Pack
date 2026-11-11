import json
import re
from comfy_api.latest import io as IO


class GenerateDictOfDicts(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateDictOfDicts_KASKI",
            display_name="Generate Dict of Dicts (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                IO.String.Input("key", default="", multiline=False),
                IO.Autogrow.Input(
                    "dicts",
                    template=IO.Autogrow.TemplatePrefix(
                        IO.Dict.Input("dict"),
                        prefix="dict",
                        min=1,
                        max=64,
                    ),
                ),
            ],
            outputs=[IO.Dict.Output(display_name="KeyValue-Dict")],
        )

    @classmethod
    def execute(
        cls,
        key: str,
        dicts: IO.Autogrow.Type,
    ) -> IO.NodeOutput:
        key = key.strip()
        if not key:
            return IO.NodeOutput({})

        merged_dict = {}
        for current_dict in dicts.values():
            if current_dict:
                merged_dict.update(current_dict)

        return IO.NodeOutput({key: merged_dict})


class IsEntryInDict(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IsEntryInDict_KASKI",
            display_name="Entry in DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                IO.Dict.Input("data"),
                IO.String.Input(
                    "key",
                    default="",
                    multiline=False,
                    tooltip="Use dots for dict keys and brackets for list indices, e.g. body_1.hooks[0].script.",
                ),
            ],
            outputs=[IO.Boolean.Output(display_name="Exists")],
        )

    @classmethod
    def execute(cls, data: dict, key: str) -> IO.NodeOutput:
        current = data

        for segment in key.strip().split("."):
            match = re.fullmatch(r"([^\[\]]+)((?:\[[0-9]+\])*)", segment)
            if match is None:
                return IO.NodeOutput(False)

            field, brackets = match.groups()
            if not isinstance(current, dict) or field not in current:
                return IO.NodeOutput(False)
            current = current[field]

            for index in re.findall(r"\[([0-9]+)\]", brackets):
                if not isinstance(current, list):
                    return IO.NodeOutput(False)
                try:
                    current = current[int(index)]
                except (IndexError, ValueError):
                    return IO.NodeOutput(False)

        return IO.NodeOutput(True)




DICTS_TO_DICTS_NODE_LIST = [
    GenerateDictOfDicts,
    IsEntryInDict,
]