import asyncio

from comfy_api.latest import IO


class AsyncDelay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        value_type = IO.MatchType.Template("value")
        return IO.Schema(
            node_id="AsyncDelay_KASKI",
            display_name="Async Delay",
            category="KASKI/async",
            inputs=[
                IO.MatchType.Input(
                    "image",
                    template=value_type,
                    display_name="value",
                    optional=True,
                    tooltip="The value to delay.",
                ),
                IO.Int.Input(
                    "delay",
                    default=1000,
                    min=0,
                    step=100,
                    tooltip="Delay in milliseconds.",
                ),
            ],
            outputs=[
                IO.MatchType.Output(template=value_type, display_name="value"),
            ],
        )

    @classmethod
    async def execute(cls, image=None, delay: int = 1000) -> IO.NodeOutput:
        await asyncio.sleep(max(0, delay) / 1000.0)
        return IO.NodeOutput(image)


ASYNC_TOOLS_NODES_LIST = [
    AsyncDelay,
]