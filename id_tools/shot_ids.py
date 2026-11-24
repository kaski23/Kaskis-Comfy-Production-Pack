"""Generate, modify and extract shot IDs for ComfyUI."""

import re

from comfy_api.latest import io

from .id_conventions import (
    CATEGORY,
    ID_COMPONENT_PATTERN,
    SHOT_STAGES,
    REGEX_SHOT_ID,
    REGEX_EXTRACT_SHOT_ID,
)


class GenerateShotID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GenerateShotID_KASKI",
            display_name="Generate Shot ID",
            category=CATEGORY,
            description="Build a shot ID from a shot number and pipeline step, with an optional project prefix and artist code.",
            inputs=[
                io.String.Input(
                    "project_name",
                    default="NONE",
                    tooltip="Optional project prefix. Use NONE or leave blank to omit it.",
                ),
                io.Int.Input(
                    "shot_no",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Numeric part of the shot ID. For example, 7 becomes sh007 with padding set to 3.",
                ),
                io.Combo.Input(
                    "pipeline_step",
                    options=SHOT_STAGES,
                    tooltip="Pipeline step to include in the shot ID.",
                ),
                io.String.Input(
                    "artist_code",
                    default="NONE",
                    tooltip="Optional artist code. Use NONE or leave blank to omit it.",
                ),
                io.Int.Input(
                    "version",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                    tooltip="Numeric version. Use -1 to output vN.",
                ),
                io.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits in the shot number. Use 0 for no padding.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="generated ID",
                    tooltip="Generated shot ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        project_name: str,
        shot_no: int,
        pipeline_step: str,
        artist_code: str,
        version: int,
        shot_no_zero_padding: int,
    ) -> io.NodeOutput:
        project_name = project_name.strip()
        artist_code = artist_code.strip()

        if (
            project_name not in ("", "NONE")
            and not re.fullmatch(ID_COMPONENT_PATTERN, project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not re.fullmatch(ID_COMPONENT_PATTERN, artist_code)
        ):
            raise ValueError(
                f"KASKI-Nodes: artist_code {artist_code} may only contain letters, numbers and hyphens."
            )

        if version != -1:
            version_string = f"v{version}"
        else:
            version_string = "vN"

        if artist_code == "" or artist_code == "NONE":
            artist_string = ""
        else:
            artist_string = f"_{artist_code}"

        shot_no_padded = f"{shot_no:0{shot_no_zero_padding}d}"

        if project_name == "" or project_name == "NONE":
            out = (
                f"sh{shot_no_padded}_{pipeline_step}"
                f"{artist_string}_{version_string}"
            )
        else:
            out = (
                f"{project_name}_sh{shot_no_padded}_{pipeline_step}"
                f"{artist_string}_{version_string}"
            )

        return io.NodeOutput(out)


class ModifyShotID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModifyShotID_KASKI",
            display_name="Modify Shot ID",
            category=CATEGORY,
            description="Update parts of an existing shot ID. Text that is not a complete shot ID passes through unchanged.",
            inputs=[
                io.String.Input(
                    "idx",
                    tooltip="Complete shot ID to modify.",
                ),
                io.String.Input(
                    "project_name",
                    default="KEEP",
                    tooltip="Use KEEP to retain the project prefix, NONE or blank to remove it, or enter a new one.",
                ),
                io.Int.Input(
                    "shot_no",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                    tooltip="New shot number. Use -1 to retain the existing shot number and its padding.",
                ),
                io.Combo.Input(
                    "pipeline_step",
                    options=["KEEP", *SHOT_STAGES],
                    tooltip="New pipeline step. Use KEEP to retain the existing one.",
                ),
                io.String.Input(
                    "artist_code",
                    default="KEEP",
                    tooltip="Use KEEP to retain the artist code, NONE or blank to remove it, or enter a new one.",
                ),
                io.Combo.Input(
                    "version",
                    options=["Keep", "Increment"],
                    tooltip="Keep the version or add 1 to a numeric version. vN stays vN.",
                ),
                io.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits when replacing the shot number. Use 0 for no padding.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="modified ID",
                    tooltip="Updated shot ID, or the original input if it is not a complete shot ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        idx: str,
        project_name: str,
        shot_no: int,
        pipeline_step: str,
        artist_code: str,
        version: str,
        shot_no_zero_padding: int,
    ) -> io.NodeOutput:
        if not REGEX_SHOT_ID.fullmatch(idx):
            return io.NodeOutput(idx)

        project_name = project_name.strip()
        artist_code = artist_code.strip()

        if (
            project_name not in ("", "NONE")
            and not re.fullmatch(ID_COMPONENT_PATTERN, project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not re.fullmatch(ID_COMPONENT_PATTERN, artist_code)
        ):
            raise ValueError(
                f"KASKI-Nodes: artist_code {artist_code} may only contain letters, numbers and hyphens."
            )

        match = REGEX_SHOT_ID.fullmatch(idx)

        if not match:
            raise ValueError(
                f"KASKI-Nodes: Invalid Shot ID: {idx}"
            )

        old_project_name = match.group("project_name") or ""
        old_shot = match.group("shot")
        old_pipeline_step = match.group("pipeline_step")
        old_artist_code = match.group("artist_code") or ""
        old_version = match.group("version")

        # --- PROJECT NAME ---
        if project_name != "KEEP":
            if project_name == "" or project_name == "NONE":
                old_project_name = ""
            else:
                old_project_name = project_name

        # --- SHOT NO ---
        if shot_no != -1:
            shot_no_padded = f"{shot_no:0{shot_no_zero_padding}d}"
            old_shot = f"sh{shot_no_padded}"

        # --- PIPELINE STEP ---
        if pipeline_step != "KEEP":
            old_pipeline_step = pipeline_step

        # --- ARTIST CODE ---
        if artist_code != "KEEP":
            if artist_code == "" or artist_code == "NONE":
                old_artist_code = ""
            else:
                old_artist_code = artist_code

        # --- VERSION ---
        if version == "Increment" and old_version != "vN":
            old_version = f"v{int(old_version[1:]) + 1}"

        # --- REBUILD ---
        if old_project_name == "":
            if old_artist_code == "":
                out = f"{old_shot}_{old_pipeline_step}_{old_version}"
            else:
                out = (
                    f"{old_shot}_{old_pipeline_step}_"
                    f"{old_artist_code}_{old_version}"
                )
        else:
            if old_artist_code == "":
                out = (
                    f"{old_project_name}_{old_shot}_"
                    f"{old_pipeline_step}_{old_version}"
                )
            else:
                out = (
                    f"{old_project_name}_{old_shot}_{old_pipeline_step}_"
                    f"{old_artist_code}_{old_version}"
                )

        return io.NodeOutput(out)


class ExtractShotID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtractShotID_KASKI",
            display_name="Extract Shot ID",
            category=CATEGORY,
            description=(
                "Find the first shot ID in text such as a filename or path. "
                "Append vN if the version is missing."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    tooltip="Text, filename or path to search for an ID.",
                ),
                io.Boolean.Input(
                    "fail_if_not_found",
                    default=True,
                    tooltip="Raise an error if no ID is found. Disable to return the original text.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="extracted ID",
                    tooltip="First matching ID, or the original text if no match is found and errors are disabled.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text: str,
        fail_if_not_found: bool,
    ) -> io.NodeOutput:
        match = REGEX_EXTRACT_SHOT_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract ID from: {text}"
                )
            else:
                return io.NodeOutput(text)

        extracted_id = match.group(0)
        if match.group("version") is None:
            extracted_id += "_vN"

        return io.NodeOutput(extracted_id)

SHOT_ID_NODES_LIST = [
    GenerateShotID,
    ModifyShotID,
    ExtractShotID,
]
