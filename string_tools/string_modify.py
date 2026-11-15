import re
import json

from comfy_api.latest import IO


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
# V3 registration
# ---------------------------------------------------------------------------

STRING_MODIFY_NODES_LIST = [
    StringSplitAtSymbol,
    JoinStrings,
]