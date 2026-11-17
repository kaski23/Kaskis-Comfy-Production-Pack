import json
import re
from comfy_api.latest import io

class GenerateJSONfromDICT(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GenerateJSONfromDICT_KASKI",
            display_name="Generate JSON from DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[io.Dict.Input("data")],
            outputs=[
                io.String.Output(display_name="JSON"),
                io.String.Output(display_name="Preview"),
            ],
        )

    @classmethod
    def execute(cls, data: dict) -> io.NodeOutput:
        if data is None:
            return io.NodeOutput("", "")

        json_string = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        pretty_json = json.dumps(data, ensure_ascii=False, indent=2)

        preview = re.sub(
            r"\\.",
            lambda match: "\n" if match[0] == r"\n" else match[0],
            pretty_json,
        )

        return io.NodeOutput(json_string, preview)


class GetValueFromDict(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetValueFromDict_KASKI",
            display_name="Get Value from DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                io.Dict.Input("data"),
                io.String.Input(
                    "key",
                    default="",
                    multiline=False,
                    tooltip="Path, e.g. body_1.data or body_1.data_field[0].",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="As String",
                    is_output_list=False,
                    tooltip="One string for the entire value. Lists are formatted as JSON array text.",
                ),
                io.Dict.Output(
                    display_name="As Dict",
                    tooltip=(
                        "The selected value under the final path segment, preserving "
                        "its original type. For 'body_1.data_field[0]', the key is 'data_field[0]'."
                    ),
                ),
                io.String.Output(
                    display_name="As String List",
                    is_output_list=True,
                    tooltip=(
                        "A ComfyUI list with one string per item, or one string "
                        "for a single value. Values are converted with str()."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, data: dict, key: str) -> io.NodeOutput:
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

        if isinstance(value, list):
            as_string = json.dumps(value, ensure_ascii=False, indent=2)
            as_string_list = [str(item) for item in value]
        else:
            as_string = str(value)
            as_string_list = [as_string]

        return io.NodeOutput(as_string, {segment: value}, as_string_list)

    @classmethod
    def fingerprint_inputs(cls, **_inputs):
        # Key cached results by the conversion and the output layout.
        output_layout = tuple(
            (output.io_type, output.is_output_list)
            for output in cls.define_schema().outputs
        )
        return "json arrays and plain values", output_layout




class GetAllValuesFromDict(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GetAllValuesFromDict_KASKI",
            display_name="Get All Values from DICT (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                io.Dict.Input("data", display_name="dict"),
                io.Boolean.Input(
                    "include_nested",
                    display_name="Include Nested",
                    default=False,
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    "values_list",
                    display_name="All values",
                    is_output_list=True,
                    tooltip="Selected values as a ComfyUI list.",
                ),
                io.String.Output(
                    "values_json",
                    display_name="All values",
                    tooltip="Selected key-value pairs as compact JSON.",
                ),
            ],
        )

    @classmethod
    def execute(cls, data: dict, include_nested: bool) -> io.NodeOutput:
        selected = data or {}

        if not include_nested:
            selected = {
                key: value
                for key, value in selected.items()
                if value is None or isinstance(value, (str, int, float, bool))
            }

        json_string = json.dumps(
            selected,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return io.NodeOutput(list(selected.values()), json_string)


DICTS_TO_JSON_NODE_LIST = [
    GenerateJSONfromDICT,
    GetValueFromDict,
    GetAllValuesFromDict,
]