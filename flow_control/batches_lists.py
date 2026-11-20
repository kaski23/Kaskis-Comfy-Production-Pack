import random

from comfy_api.latest import io
from comfy_execution.graph_utils import is_link
from nodes import NODE_CLASS_MAPPINGS


class EnumerateList_KASKI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EnumerateList_KASKI",
            display_name="Enumerate List",
            category="KASKI/flow_control",
            description="Return one zero-based integer index per input item.",
            is_input_list=True,
            inputs=[io.AnyType.Input("items")],
            outputs=[io.Int.Output(display_name="indices", is_output_list=True)],
        )

    @classmethod
    def execute(cls, items: list[object]) -> io.NodeOutput:
        return io.NodeOutput(list(range(len(items))))
        

class GetListLength_KASKI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetListLength_KASKI",
            display_name="Get List Length",
            category="KASKI/flow_control",
            description="Returns the length of any list",
            is_input_list=True,
            inputs=[io.AnyType.Input("list_input")],
            outputs=[io.Int.Output(display_name="length")],
        )

    @classmethod
    def execute(cls, list_input: list[object]) -> io.NodeOutput:
        return io.NodeOutput(len(list_input))


class EditList_KASKI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EditList_KASKI",
            display_name="Edit List",
            category="KASKI/flow_control",
            description="Resize and optionally shuffle a native ComfyUI list.",
            is_input_list=True,
            inputs=[
                io.AnyType.Input(
                    "items",
                    tooltip="Native ComfyUI list of elements of any type.",
                ),
                io.Int.Input(
                    "edit_length",
                    default=-1,
                    min=-1,
                    max=1000000,
                    step=1,
                    tooltip="Target length. -1: keep current length; 0: empty list.",
                ),
                io.Combo.Input(
                    "shorten_behavior",
                    options=["Remove_Spaced", "Remove_First", "Remove_Last"],
                    default="Remove_Spaced",
                    tooltip=(
                        "Remove_Spaced: keep evenly distributed items (including the ends "
                        "when keeping 2+). Remove_First/Last: remove from that end."
                    ),
                ),
                io.Combo.Input(
                    "lengthen_behavior",
                    options=["No_lengthening", "Repeat_First", "Repeat_Last", "Loop"],
                    default="No_lengthening",
                    tooltip=(
                        "No_lengthening: keep original length. Repeat_First/Last: "
                        "append the selected end item. Loop: repeat the input sequence."
                    ),
                ),
                io.Boolean.Input(
                    "shuffle",
                    default=False,
                    tooltip="Randomly reorder items after resizing.",
                ),
            ],
            outputs=[
                io.AnyType.Output(display_name="items", is_output_list=True),
            ],
        )

    @classmethod
    def execute(
        cls,
        items,
        edit_length,
        shorten_behavior,
        lengthen_behavior,
        shuffle,
    ) -> io.NodeOutput:
        # is_input_list=True passes all inputs as lists, including widget values.
        target = edit_length[0]
        shorten = shorten_behavior[0]
        lengthen = lengthen_behavior[0]
        do_shuffle = shuffle[0]

        result = list(items)  # never mutate the incoming list
        length = len(result)

        if target == -1:
            pass
        elif target < length:
            if target == 0:
                result = []
            elif shorten == "Remove_First":
                result = result[length - target:]
            elif shorten == "Remove_Last":
                result = result[:target]
            elif shorten == "Remove_Spaced":
                if target == 1:
                    result = result[:1]
                else:
                    # Round evenly spaced positions; preserve first and last.
                    denominator = 2 * (target - 1)
                    indices = [
                        (2 * i * (length - 1) + (target - 1)) // denominator
                        for i in range(target)
                    ]
                    result = [result[i] for i in indices]

        elif target > length and length > 0:
            missing = target - length

            if lengthen == "Repeat_First":
                result.extend([result[0]] * missing)
            elif lengthen == "Repeat_Last":
                result.extend([result[-1]] * missing)
            elif lengthen == "Loop":
                result.extend(result[i % length] for i in range(missing))

            # No_lengthening: leave unchanged.

        if do_shuffle:
            random.shuffle(result)

        return io.NodeOutput(result)

    @classmethod
    def fingerprint_inputs(cls, shuffle=None, **kwargs):
        # Connected inputs may be unknown when fingerprints are evaluated.
        # Re-run conservatively in that case to preserve shuffle behavior.
        if isinstance(shuffle, (list, tuple)):
            shuffle = shuffle[0] if shuffle else None

        return float("nan") if shuffle is None or shuffle else None



class AppendToList_KASKI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AppendToList_KASKI",
            display_name="Append to List",
            category="KASKI/flow_control",
            description="Insert a value into a native ComfyUI list.",
            is_input_list=True,
            inputs=[
                io.AnyType.Input("items"),
                io.AnyType.Input("element", tooltip="A list is inserted as a single item."),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option("First", []),
                        io.DynamicCombo.Option("Last", []),
                        io.DynamicCombo.Option(
                            "At Position",
                            [
                                io.Int.Input(
                                    "position",
                                    default=0,
                                    min=0,
                                    max=2**53 - 1,
                                    step=1,
                                    tooltip=(
                                        "Zero-based index. 0 inserts first. "
                                        "Positions past the end append."
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
            outputs=[io.AnyType.Output(display_name="items", is_output_list=True)],
            hidden=[io.Hidden.dynprompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        items: list[object],
        element: list[object],
        mode: io.DynamicCombo.Type,
    ) -> io.NodeOutput:
        if len(element) == 1 and not cls._element_is_list_output():
            value = element[0]
        else:
            value = list(element)

        selection = mode["mode"][0]
        if selection == "At Position":
            positions = mode["position"]
            if len(positions) != 1:
                raise ValueError("Append to List: supply exactly one integer for 'position'.")
            index = min(max(positions[0], 0), len(items))
        elif selection == "First":
            index = 0
        else:
            index = len(items)

        result = list(items)
        result.insert(index, value)
        return io.NodeOutput(result)

    @classmethod
    def _element_is_list_output(cls) -> bool:
        # ComfyUI wraps scalar outputs too, so a one-item list needs a schema check.
        hidden = cls.hidden
        if hidden is None or hidden.dynprompt is None:
            return False

        prompt = hidden.dynprompt
        link = prompt.get_node(hidden.unique_id)["inputs"].get("element")
        if not is_link(link):
            return False

        source_id, output_index = link
        source_node = prompt.get_node(source_id)
        source_class = NODE_CLASS_MAPPINGS[source_node["class_type"]]
        list_outputs = getattr(source_class, "OUTPUT_IS_LIST", ())
        return bool(list_outputs and list_outputs[output_index])


BATCHES_LISTS_NODES_LIST = [
    EnumerateList_KASKI,
    GetListLength_KASKI,
    EditList_KASKI,
    AppendToList_KASKI,
]
