"""Color-filled image inpainting and outpainting for ComfyUI v3."""

import torch

from comfy_api.latest import IO


class ImagesInpaintOutpaint(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_InpaintOutpaintImages",
            display_name="Inpaint/Outpaint Images with Color",
            description=(
                "Fill the input mask with a color and extend the image by percentages "
                "of its original width and height."
            ),
            category="KASKI/image",
            inputs=[
                IO.Image.Input("image"),
                IO.Float.Input("extend_top", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("extend_bottom", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("extend_left", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("extend_right", default=0.0, min=0.0, max=1000.0, step=0.1),
                IO.Color.Input("color", default="#000000"),
                IO.Mask.Input("mask", optional=True),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Mask.Output(display_name="mask"),
            ],
        )

    @classmethod
    def execute(cls, image, extend_top, extend_bottom, extend_left,
                extend_right, color, mask=None) -> IO.NodeOutput:
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
        return IO.NodeOutput(output_image, output_mask)


IMAGE_TOOLS_NODES_LIST = [
    ImagesInpaintOutpaint,
]
