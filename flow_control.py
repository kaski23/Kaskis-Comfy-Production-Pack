from comfy_api.latest import IO
from comfy_execution.graph_utils import ExecutionBlocker



# ---------------------------------------------------------------------------
# First Valid Input
# ---------------------------------------------------------------------------

class FirstValidInput_KASKI(IO.ComfyNode):

    MAX_INPUTS = 32

    @classmethod
    def define_schema(cls):

        template = IO.Autogrow.TemplateNames(
            input=IO.AnyType.Input(
                "input",
                optional=True,
            ),
            names=[
                f"input_{i}"
                for i in range(1, cls.MAX_INPUTS + 1)
            ],
            min=1,
        )

        return IO.Schema(
            node_id="FirstValidInput_KASKI",
            display_name="First Valid Input",
            category="KASKI/flow_control",

            inputs=[
                IO.Autogrow.Input(
                    "inputs",
                    template=template,
                ),
            ],

            outputs=[
                IO.AnyType.Output(
                    display_name="value",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        inputs: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:

        inputs = inputs or {}

        for i in range(1, cls.MAX_INPUTS + 1):

            value = inputs.get(f"input_{i}")

            if value is not None:
                return IO.NodeOutput(value)

        return IO.NodeOutput(None)



# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

class Gate:

    CHANNELS = 0

    @classmethod
    def define_schema(cls):

        # Independent type matching for each input/output pair
        templates = {
            i: IO.MatchType.Template(f"channel_{i}")
            for i in range(1, cls.CHANNELS + 1)
        }

        return IO.Schema(
            hidden=[IO.Hidden.unique_id],

            node_id=f"Gate{cls.CHANNELS}_KASKI",
            display_name=f"{cls.CHANNELS}-Channel Gate",
            category="KASKI/flow_control/gates",

            inputs=[
                IO.Boolean.Input(
                    "enabled",
                    default=False,
                    optional=True,
                ),

                IO.Combo.Input(
                    "behavior",
                    options=[
                        "return executionBlocker",
                        "return None",
                    ],
                    default="return None",
                    tooltip=(
                        "None: Returns None for unconnected inputs or "
                        "if turned off. Useful for optional inputs.\n"
                        "ExecutionBlocker: Disables downstream execution."
                    ),
                ),

                *[
                    IO.MatchType.Input(
                        f"input_{i}",
                        template=templates[i],
                        optional=True,
                        lazy=True,
                    )
                    for i in range(1, cls.CHANNELS + 1)
                ],
            ],

            outputs=[
                IO.MatchType.Output(
                    template=templates[i],
                    display_name=f"output_{i}",
                )
                for i in range(1, cls.CHANNELS + 1)
            ],
        )

    # -----------------------------------------------------------------------
    # Lazy Evaluation
    # -----------------------------------------------------------------------

    @classmethod
    def check_lazy_status(
        cls,
        enabled=False,
        behavior="return None",
        **inputs,
    ):

        if not enabled:
            return []

        return [
            name
            for name in (
                f"input_{i}"
                for i in range(1, cls.CHANNELS + 1)
            )
            if name in inputs and inputs[name] is None
        ]

    # -----------------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------------

    @classmethod
    def execute(
        cls,
        enabled=False,
        behavior="return None",
        **inputs,
    ) -> IO.NodeOutput:

        def fallback():
            if behavior == "return executionBlocker":
                return ExecutionBlocker(None)

            return None

        # Gate disabled: skip all channels
        if not enabled:
            return IO.NodeOutput(
                *[
                    fallback()
                    for _ in range(cls.CHANNELS)
                ]
            )

        # Gate enabled: pass each channel independently
        outputs = []

        for i in range(1, cls.CHANNELS + 1):

            value = inputs.get(f"input_{i}")

            if value is None:
                value = fallback()

            outputs.append(value)

        return IO.NodeOutput(*outputs)


class Gate1_KASKI(Gate, IO.ComfyNode):
    CHANNELS = 1

class Gate2_KASKI(Gate, IO.ComfyNode):
    CHANNELS = 2

class Gate3_KASKI(Gate, IO.ComfyNode):
    CHANNELS = 3

class Gate4_KASKI(Gate, IO.ComfyNode):
    CHANNELS = 4


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

FLOW_CONTROL_NODES_LIST = [
    FirstValidInput_KASKI,
    Gate1_KASKI,
    Gate2_KASKI,
    Gate3_KASKI,
    Gate4_KASKI,
    
]