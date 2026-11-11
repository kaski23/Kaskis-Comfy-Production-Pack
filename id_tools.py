import re

from comfy_api.latest import IO


# ID Patterns
ID_SUFFIX_PATTERN = (
    r"(?:_(?P<artist_code>[A-Za-z0-9-]+))?_"
    r"(?P<version>v(?:[0-9]+|N))"
)

# Try an explicit version first. A ComfyUI counter like _00001_.png is a
# filename suffix and can follow a versionless ID without an artist code.
EXTRACT_ID_SUFFIX_PATTERN = (
    rf"(?:{ID_SUFFIX_PATTERN}"
    r"|(?:_(?![0-9]{5,}_\.)[A-Za-z0-9-]+)?(?![A-Za-z0-9-]))"
)



# References
REFERENCE_TYPES = [
    "location",
    "prop",
    "character",
    "material",
]

REFERENCE_TYPE_PATTERN = "|".join(map(re.escape, REFERENCE_TYPES))

REFERENCE_ID_BASE_PATTERN = (
    r"(?<![A-Za-z0-9-])"
    rf"(?:(?P<project_name>(?!(?:{REFERENCE_TYPE_PATTERN})_)[A-Za-z0-9-]+)_)?"
    rf"(?P<reference_type>{REFERENCE_TYPE_PATTERN})_"
    rf"(?P<reference_name>[A-Za-z0-9-]+)"
)

REGEX_REFERENCE_ID = re.compile(REFERENCE_ID_BASE_PATTERN + ID_SUFFIX_PATTERN)
REGEX_EXTRACT_REFERENCE_ID = re.compile(
    REFERENCE_ID_BASE_PATTERN + EXTRACT_ID_SUFFIX_PATTERN
)


# Shots
SHOT_STAGES = [
    "firstFrame",
    "lastFrame",
    "referenceFrame",
    "ffToCleanup",
    "lfToCleanup",
    "rfToCleanup",
    "Depth",
    "Normal",
    "cgi",
    "Scribble",
    "plate",
    "plateToCleanup",
    "notEnhanced",
    "enhanced",
]

SHOT_STAGE_PATTERN = "|".join(map(re.escape, SHOT_STAGES))

SHOT_ID_BASE_PATTERN = (
    rf"(?:(?P<project_name>[A-Za-z0-9-]+)_)?"
    rf"(?P<shot>sh[0-9]+)_"
    rf"(?P<pipeline_step>{SHOT_STAGE_PATTERN})"
)

REGEX_SHOT_ID = re.compile(SHOT_ID_BASE_PATTERN + ID_SUFFIX_PATTERN)
REGEX_EXTRACT_SHOT_ID = re.compile(SHOT_ID_BASE_PATTERN + EXTRACT_ID_SUFFIX_PATTERN)

CATEGORY = "KASKI/ID-Tools"


# ---------------------------------------------------------------------------
# Reference ID: Generate
# ---------------------------------------------------------------------------

class GenerateReferenceID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateReferenceID_KASKI",
            display_name="Generate Reference ID",
            category=CATEGORY,
            description="Build a reference ID from a type and name, with an optional project prefix and artist code.",
            inputs=[
                IO.String.Input(
                    "project_name",
                    default="NONE",
                    tooltip=(
                        "Optional project prefix. Use NONE or leave blank to omit it. "
                        f"Reserved names are {', '.join(REFERENCE_TYPES)}."
                    ),
                ),
                IO.Combo.Input(
                    "reference_type",
                    options=REFERENCE_TYPES,
                    tooltip="The kind of asset this reference describes.",
                ),
                IO.String.Input(
                    "reference_name",
                    default="",
                    tooltip="Required asset name. Use hyphens to separate words.",
                ),
                IO.String.Input(
                    "artist_code",
                    default="NONE",
                    tooltip="Optional artist code. Use NONE or leave blank to omit it.",
                ),
                IO.Int.Input(
                    "version",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                    tooltip="Numeric version. Use -1 to output vN.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
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

        return IO.NodeOutput(out)


# ---------------------------------------------------------------------------
# Reference ID: Modify
# ---------------------------------------------------------------------------

class ModifyReferenceID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ModifyReferenceID_KASKI",
            display_name="Modify Reference ID",
            category=CATEGORY,
            description="Update the artist code or version of an existing reference ID.",
            inputs=[
                IO.String.Input(
                    "idx",
                    tooltip="Complete reference ID to modify.",
                ),
                IO.String.Input(
                    "artist_code",
                    default="KEEP",
                    tooltip="Use KEEP to retain the artist code, NONE or blank to remove it, or enter a new one.",
                ),
                IO.Combo.Input(
                    "version",
                    options=["Keep", "Increment"],
                    tooltip="Keep the version or add 1 to a numeric version. vN stays vN.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
        match = REGEX_REFERENCE_ID.fullmatch(idx)
        if not match:
            return IO.NodeOutput(idx)

        artist_code = artist_code.strip()
        if artist_code == "KEEP":
            artist_code = match.group("artist_code")
        elif artist_code in ("", "NONE"):
            artist_code = None
        elif not re.fullmatch(r"[A-Za-z0-9-]+", artist_code):
            raise ValueError(
                f"KASKI-Nodes: artist_code {artist_code} may only contain letters, numbers and hyphens."
            )

        version_string = match.group("version")
        if version == "Increment" and version_string != "vN":
            version_string = f"v{int(version_string[1:]) + 1}"

        reference_prefix = idx[:match.end("reference_name")]
        artist_string = f"_{artist_code}" if artist_code else ""
        return IO.NodeOutput(f"{reference_prefix}{artist_string}_{version_string}")


# ---------------------------------------------------------------------------
# Reference ID: Extract
# ---------------------------------------------------------------------------

class ExtractReferenceID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ExtractReferenceID_KASKI",
            display_name="Extract Reference ID",
            category=CATEGORY,
            description=(
                "Find the first reference ID in text such as a filename or path. "
                "Append vN if the version is missing."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    tooltip="Text, filename or path to search for an ID.",
                ),
                IO.Boolean.Input(
                    "fail_if_not_found",
                    default=True,
                    tooltip="Raise an error if no ID is found. Disable to return the original text.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
        match = REGEX_EXTRACT_REFERENCE_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract Reference ID from: {text}"
                )
            else:
                return IO.NodeOutput(text)

        extracted_id = match.group(0)
        if match.group("version") is None:
            extracted_id += "_vN"

        return IO.NodeOutput(extracted_id)


# ---------------------------------------------------------------------------
# Shot ID: Generate
# ---------------------------------------------------------------------------

class GenerateShotID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="GenerateShotID_KASKI",
            display_name="Generate Shot ID",
            category=CATEGORY,
            description="Build a shot ID from a shot number and pipeline step, with an optional project prefix and artist code.",
            inputs=[
                IO.String.Input(
                    "project_name",
                    default="NONE",
                    tooltip="Optional project prefix. Use NONE or leave blank to omit it.",
                ),
                IO.Int.Input(
                    "shot_no",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                    tooltip="Numeric part of the shot ID. For example, 7 becomes sh007 with padding set to 3.",
                ),
                IO.Combo.Input(
                    "pipeline_step",
                    options=SHOT_STAGES,
                    tooltip="Pipeline step to include in the shot ID.",
                ),
                IO.String.Input(
                    "artist_code",
                    default="NONE",
                    tooltip="Optional artist code. Use NONE or leave blank to omit it.",
                ),
                IO.Int.Input(
                    "version",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                    tooltip="Numeric version. Use -1 to output vN.",
                ),
                IO.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits in the shot number. Use 0 for no padding.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
        project_name = project_name.strip()
        artist_code = artist_code.strip()

        if (
            project_name not in ("", "NONE")
            and not re.fullmatch(r"[A-Za-z0-9-]+", project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not re.fullmatch(r"[A-Za-z0-9-]+", artist_code)
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

        return IO.NodeOutput(out)


# ---------------------------------------------------------------------------
# Shot ID: Modify
# ---------------------------------------------------------------------------

class ModifyShotID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ModifyShotID_KASKI",
            display_name="Modify Shot ID",
            category=CATEGORY,
            description="Update parts of an existing shot ID. Text that is not a complete shot ID passes through unchanged.",
            inputs=[
                IO.String.Input(
                    "idx",
                    tooltip="Complete shot ID to modify.",
                ),
                IO.String.Input(
                    "project_name",
                    default="KEEP",
                    tooltip="Use KEEP to retain the project prefix, NONE or blank to remove it, or enter a new one.",
                ),
                IO.Int.Input(
                    "shot_no",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                    tooltip="New shot number. Use -1 to retain the existing shot number and its padding.",
                ),
                IO.Combo.Input(
                    "pipeline_step",
                    options=["KEEP", *SHOT_STAGES],
                    tooltip="New pipeline step. Use KEEP to retain the existing one.",
                ),
                IO.String.Input(
                    "artist_code",
                    default="KEEP",
                    tooltip="Use KEEP to retain the artist code, NONE or blank to remove it, or enter a new one.",
                ),
                IO.Combo.Input(
                    "version",
                    options=["Keep", "Increment"],
                    tooltip="Keep the version or add 1 to a numeric version. vN stays vN.",
                ),
                IO.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                    tooltip="Minimum number of digits when replacing the shot number. Use 0 for no padding.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
        if not REGEX_SHOT_ID.fullmatch(idx):
            return IO.NodeOutput(idx)

        project_name = project_name.strip()
        artist_code = artist_code.strip()

        if (
            project_name not in ("", "NONE")
            and not re.fullmatch(r"[A-Za-z0-9-]+", project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not re.fullmatch(r"[A-Za-z0-9-]+", artist_code)
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

        return IO.NodeOutput(out)


# ---------------------------------------------------------------------------
# Shot ID: Extract
# ---------------------------------------------------------------------------

class ExtractShotID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ExtractShotID_KASKI",
            display_name="Extract Shot ID",
            category=CATEGORY,
            description=(
                "Find the first shot ID in text such as a filename or path. "
                "Append vN if the version is missing."
            ),
            inputs=[
                IO.String.Input(
                    "text",
                    tooltip="Text, filename or path to search for an ID.",
                ),
                IO.Boolean.Input(
                    "fail_if_not_found",
                    default=True,
                    tooltip="Raise an error if no ID is found. Disable to return the original text.",
                ),
            ],
            outputs=[
                IO.String.Output(
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
    ) -> IO.NodeOutput:
        match = REGEX_EXTRACT_SHOT_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract ID from: {text}"
                )
            else:
                return IO.NodeOutput(text)

        extracted_id = match.group(0)
        if match.group("version") is None:
            extracted_id += "_vN"

        return IO.NodeOutput(extracted_id)


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

ID_TOOLS_NODES_LIST = [
    GenerateReferenceID,
    ModifyReferenceID,
    ExtractReferenceID,
    GenerateShotID,
    ModifyShotID,
    ExtractShotID,
]
