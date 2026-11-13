import json
import os

import folder_paths
from comfy_api.latest import IO, InputImpl


def video_file_input():
    input_dir = folder_paths.get_input_directory()
    files = [
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
    ]
    files = folder_paths.filter_files_content_types(files, ["video"])
    return IO.Combo.Input(
        "file",
        options=sorted(files),
        upload=IO.UploadType.video,
    )


def video_outputs():
    return [
        IO.Video.Output(display_name="video"),
        IO.String.Output(display_name="filename"),
        IO.Image.Output(display_name="images"),
        IO.Mask.Output(display_name="alpha"),
    ]


def load_video(file: str):
    video_path = folder_paths.get_annotated_filepath(file)
    video = InputImpl.VideoFromFile(video_path)
    components = video.get_components()
    images = components.images
    alpha = components.alpha

    if alpha is not None:
        alpha = alpha[..., 0]
    else:
        alpha = images.new_ones(images.shape[:3])

    outputs = (video, os.path.basename(video_path), images, alpha)
    return outputs, components


def json_tag(tags: dict[str, str], key: str, default: list | dict) -> str:
    value = tags.get(key, "")
    try:
        decoded = json.loads(value)
    except (json.JSONDecodeError, RecursionError):
        return json.dumps(default)

    if isinstance(decoded, type(default)):
        return value
    return json.dumps(default)


class LoadVideoWithFilename(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadVideoWithFilename_KASKI",
            display_name="Load Video with Filename",
            category="KASKI/loaders",
            inputs=[video_file_input()],
            outputs=video_outputs(),
        )

    @classmethod
    def execute(cls, file: str) -> IO.NodeOutput:
        outputs, _ = load_video(file)
        return IO.NodeOutput(*outputs)

    @classmethod
    def fingerprint_inputs(cls, file: str):
        video_path = folder_paths.get_annotated_filepath(file)
        return os.path.getmtime(video_path)

    @classmethod
    def validate_inputs(cls, file: str):
        if not folder_paths.exists_annotated_filepath(file):
            return f"Invalid video file: {file}"
        return True


class LoadVideoWithKaskiMetadata(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadVideoWithKaskiMetadata_KASKI",
            display_name="Load Video with KASKI Metadata",
            category="KASKI/loaders",
            inputs=[video_file_input()],
            outputs=[
                *video_outputs(),
                IO.String.Output(
                    display_name="user_prompt",
                    tooltip="Generation prompt saved in the user_prompt tag.",
                ),
                IO.String.Output(
                    display_name="reference_files_json",
                    tooltip="JSON array of reference filenames in their saved order.",
                ),
                IO.String.Output(display_name="model_name"),
                IO.Int.Output(display_name="seed", tooltip="-1 if no valid seed was saved."),
                IO.String.Output(
                    display_name="workflow_json",
                    tooltip="Saved ComfyUI graph with layout and widgets.",
                ),
                IO.String.Output(
                    display_name="workflow_prompt_json",
                    tooltip="Saved ComfyUI API graph from the prompt tag.",
                ),
                IO.String.Output(
                    display_name="metadata_json",
                    tooltip="All container tags as stored, including any Base64 poster frame.",
                ),
            ],
        )

    @classmethod
    def execute(cls, file: str) -> IO.NodeOutput:
        outputs, components = load_video(file)
        tags = dict(components.metadata or {})

        try:
            seed = int(tags.get("seed", "-1"), 10)
        except ValueError:
            seed = -1

        return IO.NodeOutput(
            *outputs,
            tags.get("user_prompt", ""),
            json_tag(tags, "reference_files", []),
            tags.get("model_name", ""),
            seed,
            json_tag(tags, "workflow", {}),
            json_tag(tags, "prompt", {}),
            json.dumps(tags, ensure_ascii=False),
        )

    @classmethod
    def fingerprint_inputs(cls, file: str):
        video_path = folder_paths.get_annotated_filepath(file)
        stat = os.stat(video_path)
        return (stat.st_mtime_ns, stat.st_size)

    @classmethod
    def validate_inputs(cls, file: str):
        return LoadVideoWithFilename.validate_inputs(file)


VIDEO_LOADER_NODES_LIST = [
    LoadVideoWithFilename,
    LoadVideoWithKaskiMetadata,
]
