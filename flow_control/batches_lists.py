from comfy_api.latest import IO


class EnumerateList_KASKI(IO.ComfyNode):
    @classmethod
    def define_schema(cls) -> IO.Schema:
        return IO.Schema(
            node_id="EnumerateList_KASKI",
            display_name="Enumerate List",
            category="KASKI/flow_control",
            description="Return one zero-based integer index per input item.",
            is_input_list=True,
            inputs=[IO.AnyType.Input("items")],
            outputs=[IO.Int.Output(display_name="indices", is_output_list=True)],
        )

    @classmethod
    def execute(cls, items: list[object]) -> IO.NodeOutput:
        return IO.NodeOutput(list(range(len(items))))


BATCHES_LISTS_NODES_LIST = [EnumerateList_KASKI]
