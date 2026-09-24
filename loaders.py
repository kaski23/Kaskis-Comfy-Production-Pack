import hashlib
import os

import folder_paths
import nodes

from comfy_api.latest import IO, InputImpl


# ---------------------------------------------------------------------------
# Load Video + Image Sequence + Alpha
# ---------------------------------------------------------------------------

class LoadVideoWithFilename(IO.ComfyNode):

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
            ["video"],
        )

        return IO.Schema(
            node_id="LoadVideoWithFilename_KASKI",  # DO NOT CHANGE
            display_name="Load Video with Filename",
            category="KASKI/loaders",
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=sorted(files),
                    upload=IO.UploadType.video,
                ),
            ],
            outputs=[
                # Existing outputs — positions MUST stay unchanged
                IO.Video.Output(
                    display_name="video",
                ),
                IO.String.Output(
                    display_name="filename",
                ),

                # New outputs — append only
                IO.Image.Output(
                    display_name="images",
                ),
                IO.Mask.Output(
                    display_name="alpha",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        file: str,
    ) -> IO.NodeOutput:
        video_path = folder_paths.get_annotated_filepath(file)

        video = InputImpl.VideoFromFile(video_path)
        components = video.get_components()

        images = components.images
        alpha = components.alpha

        if alpha is not None:
            alpha = alpha[..., 0]
        else:
            alpha = images.new_ones(images.shape[:3])

        return IO.NodeOutput(
            video,                       # output 0 — unchanged
            os.path.basename(video_path),# output 1 — unchanged
            images,                      # output 2 — new
            alpha,                       # output 3 — new
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        file: str,
    ):
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def validate_inputs(
        cls,
        file: str,
    ):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"

        return True


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
    LoadVideoWithFilename,
    LoadImageWithFilename,
]