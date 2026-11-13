import hashlib
import os

import folder_paths
import nodes

from comfy_api.latest import IO, InputImpl


# ---------------------------------------------------------------------------
# Load Image with Filename
# ---------------------------------------------------------------------------

class LoadImageWithFilename(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()

        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f))
        ]

        files = folder_paths.filter_files_content_types(
            files,
            ["image"],
        )

        return IO.Schema(
            node_id="LoadImageWithFilename_KASKI",
            display_name="Load Image with Filename",
            category="KASKI/loaders",
            inputs=[
                IO.Combo.Input(
                    "image",
                    options=sorted(files),
                    upload=IO.UploadType.image,
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="image",
                ),
                IO.String.Output(
                    display_name="filename",
                ),
                IO.Mask.Output(
                    display_name="mask",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: str,
    ) -> IO.NodeOutput:
        image_path = folder_paths.get_annotated_filepath(image)

        # Delegate decoding/mask handling to ComfyUI core.
        core_loader = nodes.LoadImage()

        image_tensor, mask = core_loader.load_image(image)

        if image_tensor.shape[1:3] != (64, 64) and mask.shape[-2:] == (64, 64):
            mask = image_tensor.new_zeros(image_tensor.shape[:3])

        return IO.NodeOutput(
            image_tensor,
            os.path.basename(image_path),
            mask,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        image: str,
    ):
        image_path = folder_paths.get_annotated_filepath(image)

        m = hashlib.sha256()

        with open(image_path, "rb") as f:
            m.update(f.read())

        return m.digest().hex()

    @classmethod
    def validate_inputs(
        cls,
        image: str,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            return f"Invalid image file: {image}"

        return True


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

LOADERS_NODES_LIST = [
    LoadImageWithFilename,
]