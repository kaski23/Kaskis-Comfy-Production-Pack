"""Generate, modify and extract sequence IDs for ComfyUI."""

import re

from comfy_api.latest import io

from .id_conventions import (
    CATEGORY,
    ID_COMPONENT_PATTERN,
    SEQUENCE_NUMBER_PATTERN,
    SEQUENCE_STAGES,
    REGEX_SEQUENCE_ID,
    REGEX_EXTRACT_SEQUENCE_ID,
)


class GenerateSequenceID(io.ComfyNode):
    @staticmethod
    def _format_sequence_number(sequence_no: str, zero_padding: int) -> str:
        number = str(sequence_no).strip()
        if not re.fullmatch(SEQUENCE_NUMBER_PATTERN, number):
            raise ValueError(
                f"KASKI-Nodes: sequence_no {number} must be a number "
                "or a sequence-shot pair such as 100-01."
            )
        sequence_number, separator, shot_number = number.partition("-")
        sequence_number = f"{int(sequence_number):0{zero_padding}d}"
        if separator:
            shot_number = f"{int(shot_number):0{len(sequence_number)}d}"
            return f"{sequence_number}-{shot_number}"
        return sequence_number

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="GenerateSequenceID_KASKI",
            display_name="Generate Sequence ID",
            category=CATEGORY,
            description="Build a sequence ID from a sequence number and pipeline step, with an optional project prefix and artist code.",
            inputs=[
                io.String.Input(
                    "project_name",
                    default="NONE",
                    tooltip="Optional project prefix. Use NONE or leave blank to omit it.",
                ),
                io.String.Input(
                    "sequence_no",
                    default="0",
                    tooltip="Sequence number, optionally followed by a shot number: 100 or 100-01. For example, 7-01 becomes seq007-001 with padding set to 3.",
                ),
                io.Combo.Input(
                    "pipeline_step",
                    options=SEQUENCE_STAGES,
                    tooltip="Pipeline step to include in the sequence ID.",
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
                    "sequence_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits in the sequence number. Use 0 for no padding. The optional shot number uses the same width as the sequence number.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="generated ID",
                    tooltip="Generated sequence ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        project_name: str,
        sequence_no: str,
        pipeline_step: str,
        artist_code: str,
        version: int,
        sequence_no_zero_padding: int,
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

        sequence_no_padded = cls._format_sequence_number(sequence_no, sequence_no_zero_padding)

        if project_name == "" or project_name == "NONE":
            out = (
                f"seq{sequence_no_padded}_{pipeline_step}"
                f"{artist_string}_{version_string}"
            )
        else:
            out = (
                f"{project_name}_seq{sequence_no_padded}_{pipeline_step}"
                f"{artist_string}_{version_string}"
            )

        return io.NodeOutput(out)


class ModifySequenceID(io.ComfyNode):
    @staticmethod
    def _format_sequence_number(sequence_no: str, zero_padding: int) -> str:
        number = str(sequence_no).strip()
        if not re.fullmatch(SEQUENCE_NUMBER_PATTERN, number):
            raise ValueError(
                f"KASKI-Nodes: sequence_no {number} must be a number "
                "or a sequence-shot pair such as 100-01."
            )
        sequence_number, separator, shot_number = number.partition("-")
        sequence_number = f"{int(sequence_number):0{zero_padding}d}"
        if separator:
            shot_number = f"{int(shot_number):0{len(sequence_number)}d}"
            return f"{sequence_number}-{shot_number}"
        return sequence_number

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ModifySequenceID_KASKI",
            display_name="Modify Sequence ID",
            category=CATEGORY,
            description="Update parts of an existing sequence ID. Text that is not a complete sequence ID passes through unchanged.",
            inputs=[
                io.String.Input(
                    "idx",
                    tooltip="Complete sequence ID to modify.",
                ),
                io.String.Input(
                    "project_name",
                    default="KEEP",
                    tooltip="Use KEEP to retain the project prefix, NONE or blank to remove it, or enter a new one.",
                ),
                io.String.Input(
                    "sequence_no",
                    default="KEEP",
                    tooltip="New sequence number or sequence-shot pair, such as 100-01. Use KEEP or -1 to retain the existing numbers and sequence padding. The shot number is padded to match the sequence number. A plain number removes an existing shot part.",
                ),
                io.Combo.Input(
                    "pipeline_step",
                    options=["KEEP", *SEQUENCE_STAGES],
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
                    "sequence_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits when replacing the sequence number. Use 0 for no padding. The optional shot number uses the same width as the sequence number.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="modified ID",
                    tooltip="Updated sequence ID, or the original input if it is not a complete sequence ID.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        idx: str,
        project_name: str,
        sequence_no: str,
        pipeline_step: str,
        artist_code: str,
        version: str,
        sequence_no_zero_padding: int,
    ) -> io.NodeOutput:
        if not REGEX_SEQUENCE_ID.fullmatch(idx):
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

        match = REGEX_SEQUENCE_ID.fullmatch(idx)

        if not match:
            raise ValueError(
                f"KASKI-Nodes: Invalid Sequence ID: {idx}"
            )

        old_project_name = match.group("project_name") or ""
        old_sequence = match.group("sequence")
        old_pipeline_step = match.group("pipeline_step")
        old_artist_code = match.group("artist_code") or ""
        old_version = match.group("version")

        # --- PROJECT NAME ---
        if project_name != "KEEP":
            if project_name == "" or project_name == "NONE":
                old_project_name = ""
            else:
                old_project_name = project_name

        # --- SEQUENCE NO ---
        if str(sequence_no).strip() in ("KEEP", "-1"):
            sequence_number = old_sequence[3:]
            existing_padding = len(sequence_number.partition("-")[0])
            old_sequence = f"seq{cls._format_sequence_number(sequence_number, existing_padding)}"
        else:
            sequence_no_padded = cls._format_sequence_number(sequence_no, sequence_no_zero_padding)
            old_sequence = f"seq{sequence_no_padded}"

        # --- PIPELINE STEP ---
        if pipeline_step != "KEEP":
            old_pipeline_step = pipeline_step

        # --- ARTIST CODE ---
        if artist_code != "KEEP":
            if artist_code == "" or artist_code == "NONE":
                old_artist_code = ""
            else:
                old_artist_code = artist_code

        # --- VERSioN ---
        if version == "Increment" and old_version != "vN":
            old_version = f"v{int(old_version[1:]) + 1}"

        # --- REBUILD ---
        if old_project_name == "":
            if old_artist_code == "":
                out = f"{old_sequence}_{old_pipeline_step}_{old_version}"
            else:
                out = (
                    f"{old_sequence}_{old_pipeline_step}_"
                    f"{old_artist_code}_{old_version}"
                )
        else:
            if old_artist_code == "":
                out = (
                    f"{old_project_name}_{old_sequence}_"
                    f"{old_pipeline_step}_{old_version}"
                )
            else:
                out = (
                    f"{old_project_name}_{old_sequence}_{old_pipeline_step}_"
                    f"{old_artist_code}_{old_version}"
                )

        return io.NodeOutput(out)


class ExtractSequenceID(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ExtractSequenceID_KASKI",
            display_name="Extract Sequence ID",
            category=CATEGORY,
            description=(
                "Find the first sequence ID in text such as a filename or path. "
                "Discard trailing suffixes and file extensions. "
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
        match = REGEX_EXTRACT_SEQUENCE_ID.search(text)

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

SEQUENCE_ID_NODES_LIST = [
    GenerateSequenceID,
    ModifySequenceID,
    ExtractSequenceID,
]
