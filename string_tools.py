import re
import json

from comfy_api.latest import IO

# ---------------------------------------------------------------------------
# JSON String Tools
# ---------------------------------------------------------------------------

import json


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




# ---------------------------------------------------------------------------
# String Split at Symbol
# ---------------------------------------------------------------------------

class StringSplitAtSymbol(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="StringSplitAtSymbol_KASKI",
            display_name="String Split at Symbol",
            category="KASKI/stringtools",
            inputs=[
                IO.String.Input(
                    "text",
                    multiline=False,
                ),
                IO.String.Input(
                    "delimiter",
                    default="_",
                ),
                IO.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=1000,
                    step=1,
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="selected string",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text: str,
        delimiter: str,
        index: int,
    ) -> IO.NodeOutput:
        if not delimiter:
            raise ValueError(
                "KASKI-Nodes: no delimiter specified"
            )

        parts = text.split(delimiter)

        if 0 <= index < len(parts):
            return IO.NodeOutput(parts[index])

        return IO.NodeOutput("")


# ---------------------------------------------------------------------------
# Join Strings
# ---------------------------------------------------------------------------

class JoinStrings(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JoinStrings_KASKI",
            display_name="Join Strings",
            category="KASKI/stringtools",
            inputs=[
                IO.Autogrow.Input(
                    "strings",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.String.Input(
                            "string",
                            force_input=True,
                        ),
                        prefix="string_",
                        min=2,
                        max=50,
                    ),
                ),
                IO.String.Input(
                    "delimiter",
                    default="_",
                    multiline=False,
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="joined string",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        strings: IO.Autogrow.Type,
        delimiter: str,
    ) -> IO.NodeOutput:
        values = [
            value
            for value in strings.values()
            if value is not None
        ]

        return IO.NodeOutput(
            delimiter.join(values)
        )


# ---------------------------------------------------------------------------
# Number to String
# ---------------------------------------------------------------------------

class NumberToString(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NumberToString_KASKI",
            display_name="Number to String",
            category="KASKI/stringtools",
            inputs=[
                IO.Int.Input(
                    "number_int",
                ),
                IO.Float.Input(
                    "number_float",
                ),
                IO.Combo.Input(
                    "mode",
                    options=["INT", "FLOAT"],
                ),
                IO.Int.Input(
                    "zero_padding",
                    default=0,
                    min=0,
                    max=15,
                    step=1,
                ),
                IO.Int.Input(
                    "decimal_places",
                    default=2,
                    min=0,
                    max=10,
                    step=1,
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="string",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        number_int: int,
        number_float: float,
        mode: str,
        zero_padding: int,
        decimal_places: int,
    ) -> IO.NodeOutput:
        if mode == "INT":
            out = f"{number_int:0{zero_padding}d}"
        else:
            out = (
                f"{number_float:0{zero_padding}.{decimal_places}f}"
            )

        return IO.NodeOutput(out)


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

STRING_TOOLS_NODES_LIST = [
    GenerateDICTfromKV,
    GenerateDictOfDicts,
    GenerateJSONfromDICT,
    StringSplitAtSymbol,
    JoinStrings,
    NumberToString,
]