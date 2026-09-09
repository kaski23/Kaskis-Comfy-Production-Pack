import re

from comfy_api.latest import IO


# ---------------------------------------------------------------------------
# JSON String Tool
# ---------------------------------------------------------------------------

class JsonStringTool(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="JsonStringTool_KASKI",
            display_name="JSON Key-Value String",
            category="KASKI/stringtools",
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
                IO.Boolean.Input(
                    "nested",
                    default=False,
                ),
            ],
            outputs=[
                IO.String.Output(
                    display_name="JSON-String",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        key: str,
        value: str,
        nested: bool,
    ) -> IO.NodeOutput:
        key = key.strip().strip('"').rstrip(":").strip()
        value = value.strip()

        if not key:
            return IO.NodeOutput("")

        # Value absichern
        if not value:
            value = '""'
        else:
            starts_structured = value[0] in ['"', '{', '[']
            ends_structured = value[-1] in ['"', '}', ']']

            if not starts_structured:
                value = '"' + value

            if not ends_structured:
                value = value + '"'

        # Nested wrapping
        if nested:
            if not value.startswith("{"):
                value = "{\n" + value + "\n}"

        json_string = f'"{key}": {value},\n'

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
    JsonStringTool,
    StringSplitAtSymbol,
    JoinStrings,
    NumberToString,
]