import asyncio

from comfy_api.latest import IO


class AsyncDelay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="AsyncDelay_KASKI",
            display_name="Async Delay",
            category="KASKI/async",
            inputs=[
                IO.Image.Input(
                    "image",
                    tooltip="The images to delay.",
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
                IO.Image.Output(),
            ],
        )

    @classmethod
    async def execute(cls, image, delay):
        await asyncio.sleep(max(0, delay) / 1000.0)
        return IO.NodeOutput(image)
        
        
ASYNC_TOOLS_NODES_LIST = [
    AsyncDelay,
]