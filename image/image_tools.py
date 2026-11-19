"""Color-filled image inpainting and outpainting for ComfyUI v3."""

import torch
import ast
import math
import operator
import re
from functools import reduce

from simpleeval import SimpleEval

from comfy_api.latest import ComfyExtension, io



class ChannelMath(io.ComfyNode):

    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.Invert: operator.invert,
    }

    _SYNTAX = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call,
        ast.Name, ast.Load, ast.Constant, ast.Pow, *_OPERATORS,
    )

    _MATH_HELP = """VARIABLES
r1, g1, b1 are the RGB channels of image1; r2/g2/b2 belong to image2, and so on.
c1, c2, etc. automatically select the current output channel: in the R field c2=r2,
in the G field c2=g2, and in the B field c2=b2. aN reads alpha; without alpha aN=1.
lo1/hi1 are the minimum/maximum across RGB in image1, separately for each batch image.
lo2/hi2, etc. refer to the other image inputs in the same way.
x/y: positions from 0 to 1, left to right / top to bottom; w/h: width/height.
pi and e are mathematical constants. Write decimal numbers with a dot: 0.05.

OPERATORS
+ - * / // % and ** or ^ for powers. Use parentheses to control evaluation order.
Comparisons: < <= > >= == !=. Combine conditions with & and | and wrap each
comparison in parentheses: (r1 > 0.5) & (g1 < 0.2). ~ inverts a condition.

MATH
min(a,b,...), max(a,b,...): minimum/maximum per pixel. Capitalization such as Min/Max
is also accepted. abs, sign, ceil, floor, trunc, frac, round(x) / round(x,2),
sqrt, pow(x,exponent), exp, exp2, log (natural logarithm), log2, log10.
sin, cos, tan, asin, acos, atan, atan2(y,x), sinh, cosh, tanh; angles are in radians.
round uses round-half-to-even when a value is exactly halfway between two choices.
frac(x)=x-trunc(x), so the fractional part of a negative number stays negative.

IMAGE FUNCTIONS
clamp(x,lo,hi) limits values; saturate(x)=clamp(x,0,1).
mix(a,b,t) or lerp(a,b,t): a+(b-a)*t; t=0 returns a, t=1 returns b.
remap(x,inMin,inMax,outMin,outMax): linear range mapping, without clamping.
where(condition,yes,no), if(...) and iif(...): selection per pixel.
All arguments are evaluated; even an unused branch must not contain an invalid
constant-only calculation such as 1/0.
step(edge,x): 0 below the edge, otherwise 1.
smoothstep(lo,hi,x): smooth transition from 0 to 1; lo and hi must be different.
isfinite(x), isnan(x), isinf(x): diagnostic masks with results of 0 or 1.

STATISTICS
minval(r1), maxval(r1), mean(r1), sumval(r1) reduce a channel over all pixels
of each image, separately for each batch image. min/max compare values per pixel!
Use lo1/hi1 for a shared minimum/maximum across all RGB channels.

OUTPUT
All three expressions read the original inputs, not the other expressions' results.
Constants such as 0 or 0.5 fill the entire output channel.
Float32 output without automatic clipping, normalization or color management.
Values <0 and >1, as well as NaN/Inf, are preserved internally. Previews and PNG savers
may clip values; explicitly use clamp(...) in the expression when desired.
Invalid roots/logarithms and division by 0 can produce NaN/Inf.
Math expressions only: no Python statements, object attributes or indexing.
"""

    @classmethod
    def _channel_tooltip(cls, channel: str, variable: str) -> str:
        return (
            f"Expression for the {channel} output channel, evaluated at every pixel position.\n"
            f"Default: {variable} passes this channel of image1 through unchanged.\n"
            f"Examples:\n"
            f"{variable} + {variable[0]}2  — add channels from two images.\n"
            f"0.05 + 0.9 * {variable}  — map the nominal range 0–1 to 0.05–0.95.\n"
            f"clamp({variable}, 0, 1)  — hard-clip values to 0–1.\n"
            "0.2126*r1 + 0.7152*g1 + 0.0722*b1  — weighted grayscale mix.\n\n"
            + cls._MATH_HELP
        )

    @classmethod
    def _parse(cls, expression: str) -> tuple[str, ast.AST]:
        expression = " ".join(expression.strip().splitlines())
        expression = expression.replace("^", "**")
        expression = re.sub(r"\b(?:if|iif)\s*\(", "where(", expression, flags=re.IGNORECASE)
        if not expression or len(expression) > 4096:
            raise ValueError("Use one expression with 1 to 4096 characters.")
        tree = ast.parse(expression, mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > 256:
            raise ValueError("Expression is too complex; split it across nodes.")
        for node in nodes:
            if not isinstance(node, cls._SYNTAX):
                raise ValueError(
                    f"Unsupported syntax: {type(node).__name__}. "
                    "Use where(condition, yes, no), & and | for pixel conditions."
                )
            if isinstance(node, ast.Constant):
                if type(node.value) not in (int, float, bool) or abs(node.value) > 1e12:
                    raise ValueError("Only numeric constants between -1e12 and 1e12 are supported.")
                if not math.isfinite(node.value):
                    raise ValueError("Constants must be finite.")
            if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
                raise ValueError("Only named math functions are supported.")
            if isinstance(node, ast.Call):
                node.func.id = node.func.id.lower()
            if isinstance(node, ast.Compare) and len(node.ops) != 1:
                raise ValueError("Write (a < b) & (b < c) instead of a < b < c.")
        return expression, tree.body

    @classmethod
    def _evaluate(cls, parsed, context, functions):
        expression, tree = parsed
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.func.id not in functions:
                raise ValueError(f"Unknown math function '{node.func.id}'. See the channel tooltip.")
            if isinstance(node, ast.Name) and node.id not in context and node.id not in functions:
                raise ValueError(f"Unknown variable '{node.id}'. Connect its image input if needed.")
        operations = {**cls._OPERATORS, ast.Pow: functions["pow"]}
        evaluator = SimpleEval(operators=operations, functions=functions, names=context)
        return evaluator.eval(expression, previously_parsed=tree)

    @staticmethod
    def _math_functions(reference):
        def tensor(value):
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)

        def clamp(value, lower=0.0, upper=1.0):
            return torch.minimum(torch.maximum(tensor(value), tensor(lower)), tensor(upper))

        def minimum(*values):
            if not values:
                raise ValueError("min() needs at least one argument; use minval() for a channel minimum.")
            return reduce(torch.minimum, (tensor(value) for value in values))

        def maximum(*values):
            if not values:
                raise ValueError("max() needs at least one argument; use maxval() for a channel maximum.")
            return reduce(torch.maximum, (tensor(value) for value in values))

        def where(condition, yes, no):
            condition = torch.as_tensor(condition, device=reference.device).bool()
            return torch.where(condition, tensor(yes), tensor(no))

        def spatial(value, operation):
            value = tensor(value)
            return operation(value, dim=(-2, -1), keepdim=True) if value.ndim >= 2 else value

        def mix(a, b, amount):
            return tensor(a) + (tensor(b) - tensor(a)) * tensor(amount)

        def remap(value, source_min, source_max, target_min, target_max):
            return tensor(target_min) + (tensor(value) - tensor(source_min)) * (
                tensor(target_max) - tensor(target_min)
            ) / (tensor(source_max) - tensor(source_min))

        def smoothstep(lower, upper, value):
            t = clamp((tensor(value) - tensor(lower)) / (tensor(upper) - tensor(lower)))
            return t * t * (3 - 2 * t)

        def round_value(value, digits=0):
            if not isinstance(digits, (int, float)) or int(digits) != digits or not -12 <= digits <= 12:
                raise ValueError("round(x, digits) requires a scalar integer from -12 to 12.")
            return torch.round(tensor(value), decimals=int(digits))

        functions = {
            name: (lambda value, fn=fn: fn(tensor(value)))
            for name, fn in {
                "abs": torch.abs, "sqrt": torch.sqrt, "exp": torch.exp, "exp2": torch.exp2,
                "log": torch.log, "log2": torch.log2, "log10": torch.log10,
                "sin": torch.sin, "cos": torch.cos, "tan": torch.tan,
                "asin": torch.asin, "acos": torch.acos, "atan": torch.atan,
                "sinh": torch.sinh, "cosh": torch.cosh, "tanh": torch.tanh,
                "floor": torch.floor, "ceil": torch.ceil, "trunc": torch.trunc,
                "frac": torch.frac, "sign": torch.sign,
                "isfinite": torch.isfinite, "isnan": torch.isnan, "isinf": torch.isinf,
            }.items()
        }
        functions.update({
            "min": minimum,
            "max": maximum,
            "clamp": clamp,
            "saturate": lambda value: clamp(value),
            "pow": lambda base, exponent: torch.pow(tensor(base), tensor(exponent)),
            "round": round_value,
            "atan2": lambda y, x: torch.atan2(tensor(y), tensor(x)),
            "where": where,
            "mix": mix,
            "lerp": mix,
            "remap": remap,
            "step": lambda edge, value: (tensor(value) >= tensor(edge)).to(dtype=reference.dtype),
            "smoothstep": smoothstep,
            "minval": lambda value: spatial(value, torch.amin),
            "maxval": lambda value: spatial(value, torch.amax),
            "mean": lambda value: spatial(value, torch.mean),
            "sumval": lambda value: spatial(value, torch.sum),
        })
        return functions

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ChannelMath_KASKI",
            display_name="Channel Math (KASKI)",
            category="KASKI/image",
            description=(
                "Per-pixel math with freely combined RGB channels. "
                "IMAGE inputs grow automatically when connected (up to 100). "
                "image1 supplies r1/g1/b1, image2 supplies r2/g2/b2, and so on. "
                "Three expressions produce one RGB IMAGE. A detailed function reference "
                "is available in each expression tooltip. No automatic clipping."
            ),
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        input=io.Image.Input("image", tooltip=(
                            "Image source for channel variables. image1 → r1/g1/b1, "
                            "image2 → r2/g2/b2, and so on. Connecting an input adds another IMAGE socket. "
                            "The visible socket number determines the variable suffix; missing slots "
                            "are not silently renumbered by this node. All images must have the same width/height; "
                            "there is no resizing or cropping. Batch sizes must match or be 1; "
                            "a single image is repeated to match the other batch size. "
                            "RGBA inputs are accepted; aN reads alpha, but output contains RGB only."
                        )),
                        names=[f"image{i}" for i in range(1, 101)],
                        min=1,
                    ),
                    tooltip=(
                        "Automatically growing list of IMAGE sources, with image1 required. "
                        "Additional connections are addressed as image2, image3, and so on. "
                        "r1+r2 adds red at the same pixel position from image1 and image2. "
                        "Inputs are left unchanged. Calculations use Float32 on the device "
                        "of image1; other inputs are moved there if needed. "
                        "Image resolutions must match. A batch size of 1 is automatically broadcast."
                    ),
                ),
                io.String.Input("red", display_name="R – Expression", default="r1", multiline=True,
                                tooltip=cls._channel_tooltip("Red", "r1")),
                io.String.Input("green", display_name="G – Expression", default="g1", multiline=True,
                                tooltip=cls._channel_tooltip("Green", "g1")),
                io.String.Input("blue", display_name="B – Expression", default="b1", multiline=True,
                                tooltip=cls._channel_tooltip("Blue", "b1")),
            ],
            outputs=[io.Image.Output(display_name="image", tooltip=(
                "RGB IMAGE with exactly three channels: R/G/B come from the three expressions. "
                "Float32 on the device of image1, without automatic clipping or normalization. "
                "Resolution is unchanged; batch size matches the largest input batch. "
                "Alpha is not included. A constant expression fills the entire channel."
            ))],
        )

    @classmethod
    def execute(
        cls, images: io.Autogrow.Type, red="r1", green="g1", blue="b1",
    ):
        if not isinstance(images, dict):
            raise ValueError("Connect IMAGE sources to the autogrow inputs.")
        sources = {}
        for name, image in images.items():
            if image is None:
                continue
            match = re.fullmatch(r"image([1-9][0-9]*)", name)
            if not match:
                raise ValueError(f"Invalid image socket: {name!r}.")
            sources[int(match.group(1))] = image
        if 1 not in sources:
            raise ValueError("Connect an IMAGE to image1.")
        image1 = sources[1]
        for index in sorted(sources):
            image = sources[index]
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[-1] not in (3, 4):
                raise ValueError(f"image{index} must be an RGB or RGBA IMAGE tensor [B,H,W,C].")
            if min(image.shape[:3]) < 1 or image.shape[1:3] != image1.shape[1:3]:
                raise ValueError("Images must have matching width/height and non-empty batches.")
        batch = max(image.shape[0] for image in sources.values())
        if any(image.shape[0] not in (1, batch) for image in sources.values()):
            raise ValueError("Image batches must match or contain one image for broadcasting.")
        expressions = [red, green, blue]
        parsed = []
        for channel, expression in zip("RGB", expressions):
            try:
                parsed.append(cls._parse(expression))
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"{channel} expression: {error}") from error
        used = {node.id for _, tree in parsed for node in ast.walk(tree) if isinstance(node, ast.Name)}
        height, width = image1.shape[1:3]
        sources = {
            index: image.to(device=image1.device, dtype=torch.float32).expand(batch, -1, -1, -1)
            for index, image in sources.items()
        }
        reference = sources[1][..., 0]
        context = dict(w=width, h=height, pi=math.pi, e=math.e)
        if "x" in used:
            context["x"] = torch.linspace(0, 1, width, device=reference.device, dtype=reference.dtype).view(1, 1, width)
        if "y" in used:
            context["y"] = torch.linspace(0, 1, height, device=reference.device, dtype=reference.dtype).view(1, height, 1)
        channels = {}
        for index, image in sources.items():
            planes = [image[..., i] for i in range(3)]
            channels[index] = planes
            context.update({f"{name}{index}": plane for name, plane in zip("rgb", planes)})
            if f"a{index}" in used:
                context[f"a{index}"] = image[..., 3] if image.shape[-1] == 4 else 1.0
            if f"lo{index}" in used:
                context[f"lo{index}"] = image[..., :3].amin(dim=(1, 2, 3)).view(batch, 1, 1)
            if f"hi{index}" in used:
                context[f"hi{index}"] = image[..., :3].amax(dim=(1, 2, 3)).view(batch, 1, 1)
        functions = cls._math_functions(reference)
        result = []
        for channel, expression in enumerate(parsed):
            context.update({f"c{index}": planes[channel] for index, planes in channels.items()})
            try:
                value = cls._evaluate(expression, context, functions)
                value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
                result.append(torch.broadcast_to(value, reference.shape))
            except (ValueError, TypeError, ArithmeticError, RuntimeError) as error:
                raise ValueError(f"{'RGB'[channel]} expression: {error}") from error
        return io.NodeOutput(torch.stack(result, dim=-1))




class ImagesInpaintOutpaint(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="KASKI_InpaintOutpaintImages",
            display_name="Inpaint/Outpaint Images with Color",
            description=(
                "Fill the input mask with a color and extend the image by percentages "
                "of its original width and height."
            ),
            category="KASKI/image",
            inputs=[
                io.Image.Input("image"),
                io.Float.Input("extend_top", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Float.Input("extend_bottom", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Float.Input("extend_left", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Float.Input("extend_right", default=0.0, min=0.0, max=1000.0, step=0.1),
                io.Color.Input("color", default="#000000"),
                io.Mask.Input("mask", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, extend_top, extend_bottom, extend_left,
                extend_right, color, mask=None) -> io.NodeOutput:
        if image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("IMAGE must have shape [B, H, W, 3].")

        batch, height, width, _ = image.shape
        if batch < 1 or height < 1 or width < 1:
            raise ValueError("IMAGE must contain at least one nonempty image.")

        top, bottom = (int(height * float(p) / 100 + 0.5)
                       for p in (extend_top, extend_bottom))
        left, right = (int(width * float(p) / 100 + 0.5)
                       for p in (extend_left, extend_right))
        if min(top, bottom, left, right) < 0:
            raise ValueError("Extension percentages must be nonnegative.")

        hex_color = color.lstrip("#")
        if len(hex_color) != 6:
            raise ValueError("Color must be a six-digit hex RGB value, e.g. #ff8000.")
        try:
            rgb = [int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        except ValueError as exc:
            raise ValueError("Color must be a six-digit hex RGB value.") from exc

        fill = image.new_tensor(rgb).view(1, 1, 1, 3)
        if mask is None or mask.numel() == 0:
            source_mask = image.new_zeros((batch, height, width))
            mask_dtype = image.dtype
        else:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim != 3 or mask.shape[1:] != (height, width):
                raise ValueError("MASK must have shape [B, H, W] matching IMAGE height and width.")
            mask_dtype = mask.dtype
            source_mask = mask.to(device=image.device, dtype=image.dtype).clamp(0, 1)
            mask_count = source_mask.shape[0]
            if mask_count < batch:
                source_mask = torch.cat(
                    (source_mask, source_mask[-1:].expand(batch - mask_count, -1, -1)),
                    dim=0,
                )
            elif mask_count > batch:
                source_mask = source_mask[:batch]

        alpha = source_mask.unsqueeze(-1)
        center = image * (1 - alpha) + fill * alpha

        output_image = torch.empty(
            (batch, height + top + bottom, width + left + right, 3),
            dtype=image.dtype, device=image.device,
        )
        output_image[:] = fill
        output_image[:, top:top + height, left:left + width, :] = center

        output_mask = torch.ones(
            (batch, height + top + bottom, width + left + right),
            dtype=mask_dtype, device=image.device,
        )
        output_mask[:, top:top + height, left:left + width] = source_mask.to(mask_dtype)
        return io.NodeOutput(output_image, output_mask)
        
        
        
class CombineImagesToImageList(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        image_inputs = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"), prefix="image", min=1, max=50
        )
        return io.Schema(
            node_id="KASKI_ImagesToList",
            display_name="Create Image List",
            category="KASKI/image",
            description="Collects images in input order, preserving their sizes. "
            "Each frame of an input batch becomes a separate list item.",
            is_input_list=True,
            inputs=[io.Autogrow.Input("images", template=image_inputs)],
            outputs=[io.Image.Output("images", is_output_list=True)],
        )

    @classmethod
    def execute(cls, images: dict[str, list[io.Image.Type]]) -> io.NodeOutput:
        image_list = []
        for image_batches in images.values():
            for batch in image_batches:
                for index in range(batch.shape[0]):
                    image_list.append(batch[index : index + 1])
        return io.NodeOutput(image_list)




IMAGE_TOOLS_NODES_LIST = [
    ChannelMath,
    ImagesInpaintOutpaint,
    CombineImagesToImageList,
]