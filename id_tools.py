
import re

from comfy_api.latest import IO


# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

REGEX_REFERENCE_ID = re.compile(
    r"(?:[A-Za-z0-9-]+_)?(?:character|prop|location|material)_[A-Za-z0-9-]+_(?:[A-Za-z0-9-]+_)?v(?:[0-9]+|N)"
)

SHOT_STAGES = [
    "firstFrame",
    "lastFrame",
    "ffToCleanup",
    "lfToCleanup",
    "Depth",
    "Normal",
    "cgi",
    "Scribble",
    "plate",
    "plateToCleanup",
    "notEnhanced",
    "enhanced",
]

STAGE_PATTERN = "|".join(map(re.escape, SHOT_STAGES))

REGEX_ID = re.compile(
    rf"(?:(?P<project_name>[A-Za-z0-9-]+)_)?"
    rf"(?P<shot>sh[0-9]+)_"
    rf"(?P<pipeline_step>{STAGE_PATTERN})"
    rf"(?:_(?P<artist_code>[A-Za-z0-9-]+))?_"
    rf"(?P<version>v(?:[0-9]+|N))"
)

ID_COMPONENT_REGEX = re.compile(r"[A-Za-z0-9-]+")

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
            inputs=[
                IO.String.Input("project_name", default="NONE"),
                IO.Combo.Input(
                    "reference_type",
                    options=["character", "prop", "location", "material"],
                ),
                IO.String.Input("reference_name", default=""),
                IO.String.Input("artist_code", default="NONE"),
                IO.Int.Input(
                    "version",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="generated ID"),
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
# Reference ID: Extract
# ---------------------------------------------------------------------------

class ExtractReferenceID(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ExtractReferenceID_KASKI",
            display_name="Extract Reference ID",
            category=CATEGORY,
            inputs=[
                IO.String.Input("text"),
                IO.Boolean.Input("fail_if_not_found", default=True),
            ],
            outputs=[
                IO.String.Output(display_name="extracted ID"),
            ],
        )

    @classmethod
    def execute(
        cls,
        text: str,
        fail_if_not_found: bool,
    ) -> IO.NodeOutput:
        match = REGEX_REFERENCE_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract Reference ID from: {text}"
                )
            else:
                return IO.NodeOutput(text)

        return IO.NodeOutput(match.group(0))


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
            inputs=[
                IO.String.Input("project_name", default="NONE"),
                IO.Int.Input(
                    "shot_no",
                    default=0,
                    min=0,
                    max=10000,
                    step=1,
                ),
                IO.Combo.Input(
                    "pipeline_step",
                    options=SHOT_STAGES,
                ),
                IO.String.Input("artist_code", default="NONE"),
                IO.Int.Input(
                    "version",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                ),
                IO.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="generated ID"),
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
            and not ID_COMPONENT_REGEX.fullmatch(project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not ID_COMPONENT_REGEX.fullmatch(artist_code)
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
            inputs=[
                IO.String.Input("idx"),
                IO.String.Input("project_name", default="KEEP"),
                IO.Int.Input(
                    "shot_no",
                    default=-1,
                    min=-1,
                    max=10000,
                    step=1,
                ),
                IO.Combo.Input(
                    "pipeline_step",
                    options=["KEEP", *SHOT_STAGES],
                ),
                IO.String.Input("artist_code", default="KEEP"),
                IO.Combo.Input(
                    "version",
                    options=["Keep", "Increment"],
                ),
                IO.Int.Input(
                    "shot_no_zero_padding",
                    default=3,
                    min=0,
                    max=15,
                    step=1,
                ),
            ],
            outputs=[
                IO.String.Output(display_name="modified ID"),
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
        if not REGEX_ID.fullmatch(idx):
            return IO.NodeOutput(idx)

        project_name = project_name.strip()
        artist_code = artist_code.strip()

        if (
            project_name not in ("", "NONE")
            and not ID_COMPONENT_REGEX.fullmatch(project_name)
        ):
            raise ValueError(
                f"KASKI-Nodes: project_name {project_name} may only contain letters, numbers and hyphens."
            )

        if (
            artist_code not in ("", "NONE")
            and not ID_COMPONENT_REGEX.fullmatch(artist_code)
        ):
            raise ValueError(
                f"KASKI-Nodes: artist_code {artist_code} may only contain letters, numbers and hyphens."
            )

        match = REGEX_ID.fullmatch(idx)

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
            inputs=[
                IO.String.Input("text"),
                IO.Boolean.Input("fail_if_not_found", default=True),
            ],
            outputs=[
                IO.String.Output(display_name="extracted ID"),
            ],
        )

    @classmethod
    def execute(
        cls,
        text: str,
        fail_if_not_found: bool,
    ) -> IO.NodeOutput:
        match = REGEX_ID.search(text)

        if not match:
            if fail_if_not_found:
                raise ValueError(
                    f"KASKI-Nodes: Couldn't extract ID from: {text}"
                )
            else:
                return IO.NodeOutput(text)

        return IO.NodeOutput(match.group(0))


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

ID_TOOLS_NODE_LIST = [
    GenerateReferenceID,
    ExtractReferenceID,
    GenerateShotID,
    ModifyShotID,
    ExtractShotID,
]