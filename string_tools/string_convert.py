import re
import json
from string import Formatter

from comfy_api.latest import ComfyExtension, io


# ---------------------------------------------------------------------------
# Any to String
# ---------------------------------------------------------------------------

class AnyToString(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AnyToString_KASKI",
            display_name="Any to String",
            category="KASKI/stringtools",
            description="Convert any value to text. Lists, tuples, and dictionaries use JSON when possible.",
            inputs=[io.AnyType.Input("value")],
            outputs=[io.String.Output(display_name="string")],
        )

    @classmethod
    def execute(cls, value: object) -> io.NodeOutput:
        if isinstance(value, (dict, list, tuple)):
            try:
                text = json.dumps(value, ensure_ascii=False, allow_nan=False, default=cls._as_text)
            except Exception:
                pass
            else:
                return io.NodeOutput(text)

        return io.NodeOutput(cls._as_text(value))
    
    @staticmethod
    def _as_text(value: object) -> str:
        if isinstance(value, str):
            return value

        if isinstance(value, (bytes, bytearray)):
            try:
                return value.decode("utf-8")
            except Exception:
                pass

        for formatter in (str, repr):
            try:
                return formatter(value)
            except Exception:
                continue

        return f"<{type(value).__name__}>"



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
    AnyToString,
    FString,
]