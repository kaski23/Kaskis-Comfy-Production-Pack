import hashlib
import os

import folder_paths
import nodes

from comfy_api.latest import IO, InputImpl


# ---------------------------------------------------------------------------
# Load Video with Filename
# ---------------------------------------------------------------------------

class LoadVideoWithFilename(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()

        os.makedirs(
            input_dir,
            exist_ok=True,
        )

        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(
                os.path.join(input_dir, f)
            )
        ]

        files = folder_paths.filter_files_content_types(
            files,
            ["video"],
        )

        return IO.Schema(
            node_id="LoadVideoWithFilename_KASKI",
            display_name="Load Video with Filename",
            category="KASKI/loaders",
            search_aliases=[
                "load video filename",
                "video filename",
                "import video",
            ],
            inputs=[
                IO.Combo.Input(
                    "file",
                    options=sorted(files),
                    upload=IO.UploadType.video,
                ),
            ],
            outputs=[
                IO.Video.Output(
                    display_name="video",
                ),
                IO.String.Output(
                    display_name="filename",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        file: str,
    ) -> IO.NodeOutput:

        video_path = folder_paths.get_annotated_filepath(
            file
        )

        video = InputImpl.VideoFromFile(
            video_path
        )

        filename = os.path.basename(
            video_path
        )

        return IO.NodeOutput(
            video,
            filename,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        file: str,
    ):
        video_path = folder_paths.get_annotated_filepath(
            file
        )

        # Mirrors current core LoadVideo:
        # avoid hashing potentially huge video files.
        return os.path.getmtime(
            video_path
        )

    @classmethod
    def validate_inputs(
        cls,
        file: str,
    ):
        if not folder_paths.exists_annotated_filepath(
            file
        ):
            return f"Invalid video file: {file}"

        return True


# ---------------------------------------------------------------------------
# Load Image with Filename
# ---------------------------------------------------------------------------

class LoadImageWithFilename(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()

        os.makedirs(
            input_dir,
            exist_ok=True,
        )

        files = [
            f
            for f in os.listdir(input_dir)
            if os.path.isfile(
                os.path.join(input_dir, f)
            )
        ]

        files = folder_paths.filter_files_content_types(
            files,
            ["image"],
        )

        return IO.Schema(
            node_id="LoadImageWithFilename_KASKI",
            display_name="Load Image with Filename",
            category="KASKI/loaders",
            search_aliases=[
                "load image filename",
                "image filename",
                "import image",
            ],
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
            ],
        )

    @classmethod
    def execute(
        cls,
        image: str,
    ) -> IO.NodeOutput:

        image_path = folder_paths.get_annotated_filepath(
            image
        )

        # Delegate image decoding entirely to ComfyUI's core loader.
        #
        # Current LoadImage returns:
        #   IMAGE, MASK
        #
        # We only need IMAGE.
        core_loader = nodes.LoadImage()

        loaded = core_loader.load_image(
            image
        )

        image_tensor = loaded[0]

        filename = os.path.basename(
            image_path
        )

        return IO.NodeOutput(
            image_tensor,
            filename,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        image: str,
    ):
        image_path = folder_paths.get_annotated_filepath(
            image
        )

        m = hashlib.sha256()

        with open(
            image_path,
            "rb",
        ) as f:
            m.update(
                f.read()
            )

        return m.digest().hex()

    @classmethod
    def validate_inputs(
        cls,
        image: str,
    ):
        if not folder_paths.exists_annotated_filepath(
            image
        ):
            return f"Invalid image file: {image}"

        return True


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

LOADERS_NODES_LIST = [
    LoadVideoWithFilename,
    LoadImageWithFilename,
]