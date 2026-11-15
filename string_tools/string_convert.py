import re
import json
from string import Formatter

from comfy_api.latest import ComfyExtension, io


# ---------------------------------------------------------------------------
# Number to String
# ---------------------------------------------------------------------------

class NumberToString(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="NumberToString_KASKI",
            display_name="Number to String",
            category="KASKI/stringtools",
            inputs=[
                io.Int.Input(
                    "number_int",
                ),
                io.Float.Input(
                    "number_float",
                ),
                io.Combo.Input(
                    "mode",
                    options=["INT", "FLOAT"],
                ),
                io.Int.Input(
                    "zero_padding",
                    default=0,
                    min=0,
                    max=15,
                    step=1,
                ),
                io.Int.Input(
                    "decimal_places",
                    default=2,
                    min=0,
                    max=10,
                    step=1,
                ),
            ],
            outputs=[
                io.String.Output(
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
    ) -> io.NodeOutput:
        if mode == "INT":
            out = f"{number_int:0{zero_padding}d}"
        else:
            out = (
                f"{number_float:0{zero_padding}.{decimal_places}f}"
            )

        return io.NodeOutput(out)



class FString(io.ComfyNode, Formatter):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FStringAutogrow_KASKI",
            display_name="FString Input",
            category="KASKI/string",
            inputs=[
                io.String.Input(
                    "template",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Insert connected values with {input_0}, {input_1}, etc. "
                        "Use {{ and }} for literal braces."
                    ),
                ),
                io.Autogrow.Input(
                    "values",
                    template=io.Autogrow.TemplatePrefix(
                        io.AnyType.Input("value"),
                        prefix="input_",
                        min=0,
                        max=100,
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="String",
                    tooltip="The template with connected values substituted as text.",
                ),
            ],
        )

    @classmethod
    def execute(cls, template: str, values: io.Autogrow.Type) -> io.NodeOutput:
        return io.NodeOutput(cls().vformat(template, (), values))

    def get_field(self, field_name: str, args, kwargs):
        if not re.fullmatch(r"input_[0-9]+", field_name):
            raise ValueError(
                f"Invalid placeholder {field_name!r}. Use {{input_0}}, {{input_1}}, etc."
            )
        if field_name not in kwargs:
            raise ValueError(
                f"Connect {field_name} to use {{{field_name}}} in the template."
            )
        return kwargs[field_name], field_name
# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

STRING_CONVERT_NODES_LIST = [
    NumberToString,
    FString,
]