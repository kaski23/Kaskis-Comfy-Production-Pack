import json
import re
from comfy_api.latest import io as IO


class GenerateDICTfromKV(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateDICTfromKV_KASKI",
            display_name="Generate DICT from Key-Value-Pair (JSON-Tools)",
            category="KASKI/jsontools",

            inputs=[
                IO.String.Input(
                    "key",
                    default="",
                    multiline=False,
                ),

                IO.String.Input(
                    "value",
                    default="",
                    multiline=True,
                ),
            ],

            outputs=[
                IO.Dict.Output(
                    display_name="KeyValue-Dict",
                ),
            ],
        )

    @staticmethod
    def parse_json_or_string(value: str):
        value = value.strip()

        if not value:
            return value

        # 1. Already valid standalone JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

        fragment = value.rstrip(",")

        # 2. JSON key-value fragment -> DICT
        try:
            parsed = json.loads(
                "{\n" + fragment + "\n}"
            )

            if isinstance(parsed, dict):
                return parsed

        except json.JSONDecodeError:
            pass

        # 3. JSON value fragment -> LIST
        try:
            parsed = json.loads(
                "[\n" + fragment + "\n]"
            )

            if isinstance(parsed, list):
                return parsed

        except json.JSONDecodeError:
            pass

        # 4. Normal string
        return value

    @classmethod
    def execute(
            cls,
            key: str,
            value: str,
    ) -> IO.NodeOutput:
        key = key.strip()
        if not key:
            return IO.NodeOutput({})

        if value == "" or value is None:
            return IO.NodeOutput({})

        return IO.NodeOutput({key: cls.parse_json_or_string(value)})


class GenerateDICTfromJSON(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateDICTfromJSON_KASKI",
            display_name="Generate DICT from JSON (JSON-Tools)",
            category="KASKI/jsontools",
            inputs=[
                IO.String.Input(
                    "json_text",
                    default="",
                    multiline=True,
                    tooltip="A JSON object, optionally inside a Markdown code fence.",
                ),
            ],
            outputs=[IO.Dict.Output(display_name="DICT")],
        )

    @classmethod
    def execute(cls, json_text: str) -> IO.NodeOutput:
        text = json_text.lstrip().removeprefix("\ufeff").strip()

        fence = re.fullmatch(
            r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if fence:
            text = fence.group(1).lstrip().removeprefix("\ufeff").strip()

        invalid_constants = []
        try:
            data = json.loads(text, parse_constant=invalid_constants.append)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON (line {exc.lineno}, column {exc.colno}): {exc.msg}"
            ) from exc

        if invalid_constants:
            raise ValueError(f"Invalid JSON constant: {invalid_constants[0]}.")

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the root (an object in {}).")

        return IO.NodeOutput(data)


JSON_TO_DICT_NODE_LIST = [
    GenerateDICTfromKV,
    GenerateDICTfromJSON,
]