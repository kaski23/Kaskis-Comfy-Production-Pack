"""Generate, modify and extract reference IDs for ComfyUI."""

import re

from comfy_api.latest import io

from .id_conventions import (
    CATEGORY,
    ID_COMPONENT_PATTERN,
    REFERENCE_TYPES,
    REGEX_REFERENCE_ID,
    REGEX_EXTRACT_REFERENCE_ID,
)


class GenerateReferenceID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GenerateReferenceID_KASKI",
            display_name="Generate Reference ID",
            category=CATEGORY,
            description="Build a reference ID from a type and name, with an optional project prefix and artist code.",
            inputs=[
                io.String.Input(
                    "project_name",
                    default="NONE",
                    tooltip=(
                        "Optional project prefix. Use NONE or leave blank to omit it. "
                        f"Reserved names are {', '.join(REFERENCE_TYPES)}."
                    ),
                ),
                io.Combo.Input(
                    "reference_type",
                    options=REFERENCE_TYPES,
                    tooltip="The kind of asset this reference describes.",
                ),
                io.String.Input(
                    "reference_name",
                    default="",
                    tooltip="Required asset name. Use hyphens to separate words.",
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
            ],
            outputs=[
                io.String.Output(
                    display_name="generated ID",
                    tooltip="Generated reference ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        project_name: str,
        reference_type: str,
        reference_name: str,
        artist_code: str,
        version: int,
    ) -> io.NodeOutput:
        project_name = project_name.strip()
        reference_name = reference_name.strip()
        artist_code = artist_code.strip()

        if project_name in REFERENCE_TYPES:
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} is reserved as a reference type."
            )

        if "_" in project_name:
            raise ValueError(
                f"KASKI-Nodes: project_name must not contain underscores: {project_name}"
            )

        if "_" in reference_name:
            raise ValueError(
                f"KASKI-Nodes: reference_name must not contain underscores: {reference_name}"
            )

        if "_" in artist_code:
            raise ValueError(
                f"KASKI-Nodes: artist_code must not contain underscores: {artist_code}"
            )

        if " " in project_name:
            raise ValueError(
                f"KASKI-Nodes: project_name must not contain spaces: {project_name}"
            )

        if " " in reference_name:
            raise ValueError(
                f"KASKI-Nodes: reference_name must not contain spaces: {reference_name}"
            )

        if " " in artist_code:
            raise ValueError(
                f"KASKI-Nodes: artist_code must not contain spaces: {artist_code}"
            )

        if reference_name == "":
            raise ValueError("KASKI-Nodes: reference_name cannot be empty")

        if version != -1:
            version_string = f"v{version}"
        else:
            version_string = "vN"

        if artist_code == "" or artist_code == "NONE":
            artist_string = ""
        else:
            artist_string = f"_{artist_code}"

        if project_name == "" or project_name == "NONE":
            out = (
                f"{reference_type}_{reference_name}"
                f"{artist_string}_{version_string}"
            )
        else:
            out = (
                f"{project_name}_{reference_type}_{reference_name}"
                f"{artist_string}_{version_string}"
            )

        return io.NodeOutput(out)


class ModifyReferenceID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModifyReferenceID_KASKI",
            display_name="Modify Reference ID",
            category=CATEGORY,
            description="Update the artist code or version of an existing reference ID.",
            inputs=[
                io.String.Input(
                    "idx",
                    tooltip="Complete reference ID to modify.",
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
            ],
            outputs=[
                io.String.Output(
                    display_name="modified ID",
                    tooltip="Updated reference ID, or the original input if it is not a complete reference ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        idx: str,
        artist_code: str,
        version: str,
    ) -> io.NodeOutput:
        match = REGEX_REFERENCE_ID.fullmatch(idx)
        if not match:
            return io.NodeOutput(idx)

        artist_code = artist_code.strip()
        if artist_code == "KEEP":
            artist_code = match.group("artist_code")
        elif artist_code in ("", "NONE"):
            artist_code = None
        elif not re.fullmatch(ID_COMPONENT_PATTERN, artist_code):
            raise ValueError(
                f"KASKI-Nodes: artist_code {artist_code} may only contain letters, numbers and hyphens."
            )

        version_string = match.group("version")
        if version == "Increment" and version_string != "vN":
            version_string = f"v{int(version_string[1:]) + 1}"

        reference_prefix = idx[:match.end("reference_name")]
        artist_string = f"_{artist_code}" if artist_code else ""
        return io.NodeOutput(f"{reference_prefix}{artist_string}_{version_string}")


class ExtractReferenceID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtractReferenceID_KASKI",
            display_name="Extract Reference ID",
            category=CATEGORY,
            description=(
                "Find the first reference ID in text such as a filename or path. "
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
        match = REGEX_EXTRACT_REFERENCE_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract Reference ID from: {text}"
                )
            else:
                return io.NodeOutput(text)

        extracted_id = match.group(0)
        if match.group("version") is None:
            extracted_id += "_vN"

        return io.NodeOutput(extracted_id)

REFERENCE_ID_NODES_LIST = [
    GenerateReferenceID,
    ModifyReferenceID,
    ExtractReferenceID,
]
