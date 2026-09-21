import math

import torch
import torch.nn.functional as F

from comfy_api.latest import IO
from comfy.utils import ProgressBar

from .external_libraries.RIFE import (
    calculate_optical_flow,
    interpolate_between_two_frames,
)


# ---------------------------------------------------------------------------
# Extend Video
# ---------------------------------------------------------------------------

class ExtendVideo(IO.ComfyNode):
    """
    Extends an IMAGE batch interpreted as a video sequence.

    If the input already contains at least n_frames, it is returned
    unchanged.

    The "interpolated" method temporally stretches the complete input
    sequence to exactly n_frames using RIFE interpolation.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ExtendVideo_KASKI",
            display_name="Extend Video",
            category="KASKI/videoTools",
            inputs=[
                IO.Image.Input(
                    "video",
                    tooltip=(
                        "Input image sequence to extend if it contains fewer "
                        "than the requested number of frames."
                    ),
                ),
                IO.Int.Input(
                    "n_frames",
                    default=25,
                    min=1,
                    max=99999,
                    tooltip="Minimum number of frames the output sequence should contain.",
                ),
                IO.Combo.Input(
                    "method",
                    options=[
                        "ping_pong",
                        "repeat_last_frame",
                        "repeat_from_start",
                        "interpolated",
                    ],
                    default="ping_pong",
                    tooltip=(
                        "How additional frames are generated: reverse the sequence, "
                        "hold the last frame, loop from the beginning, or temporally "
                        "stretch the sequence using RIFE interpolation."
                    ),
                ),
                # UI-Logik:
                # nur anzeigen, wenn method == "interpolated"
                IO.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.05,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "RIFE internal scale used by the interpolated method. "
                        "Lower values increase internal downscaling and often work "
                        "better for UHD footage."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="video",
                    tooltip="Input sequence extended to at least the requested frame count.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: torch.Tensor,
        n_frames: int,
        method: str,
        scale: float,
    ) -> IO.NodeOutput:

        if video.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), "
                f"got {tuple(video.shape)}"
            )

        current_frames = video.shape[0]

        if current_frames == 0:
            raise ValueError("Video contains no frames.")

        if current_frames >= n_frames:
            return IO.NodeOutput(video)

        if method == "interpolated":
            output = cls._extend_interpolated(
                video,
                n_frames,
                scale,
            )

            return IO.NodeOutput(
                output.contiguous()
            )

        missing_frames = n_frames - current_frames

        if method == "ping_pong":
            extension = cls._extend_ping_pong(
                video,
                missing_frames,
            )

        elif method == "repeat_last_frame":
            extension = cls._extend_last_frame(
                video,
                missing_frames,
            )

        elif method == "repeat_from_start":
            extension = cls._extend_from_start(
                video,
                missing_frames,
            )

        else:
            raise ValueError(
                f"Unknown extension method: {method}"
            )

        output = torch.cat(
            (video, extension),
            dim=0,
        )

        return IO.NodeOutput(
            output.contiguous()
        )

    @staticmethod
    def _extend_last_frame(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        return video[-1:].repeat(
            count,
            1,
            1,
            1,
        )

    @staticmethod
    def _extend_from_start(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        repetitions = math.ceil(
            count / video.shape[0]
        )

        return video.repeat(
            repetitions,
            1,
            1,
            1,
        )[:count]

    @staticmethod
    def _extend_ping_pong(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        frame_count = video.shape[0]

        if frame_count == 1:
            return video.repeat(
                count,
                1,
                1,
                1,
            )

        reverse = video[:-1].flip(0)
        forward = video[1:]

        cycle = torch.cat(
            (reverse, forward),
            dim=0,
        )

        repetitions = math.ceil(
            count / cycle.shape[0]
        )

        return cycle.repeat(
            repetitions,
            1,
            1,
            1,
        )[:count]

    @staticmethod
    def _extend_interpolated(
        video: torch.Tensor,
        target_frames: int,
        scale: float,
    ) -> torch.Tensor:
        source_frames = video.shape[0]

        if source_frames == 1:
            return video.repeat(
                target_frames,
                1,
                1,
                1,
            )

        positions = torch.linspace(
            0.0,
            source_frames - 1,
            steps=target_frames,
            device=video.device,
        )

        output_frames = []

        generated_frames = 0

        for position in positions:
            source_position = float(position.item())

            lower_index = math.floor(source_position)
            upper_index = min(
                lower_index + 1,
                source_frames - 1,
            )

            timestep = (
                source_position
                - lower_index
            )

            if (
                lower_index != upper_index
                and timestep > 1e-8
                and timestep < 1.0 - 1e-8
            ):
                generated_frames += 1

        progress_bar = ProgressBar(
            generated_frames
        )

        for position in positions:
            source_position = float(position.item())

            lower_index = math.floor(source_position)
            upper_index = min(
                lower_index + 1,
                source_frames - 1,
            )

            timestep = (
                source_position
                - lower_index
            )

            if (
                lower_index == upper_index
                or timestep <= 1e-8
            ):
                frame = video[lower_index]

            elif timestep >= 1.0 - 1e-8:
                frame = video[upper_index]

            else:
                frame = interpolate_between_two_frames(
                    video[lower_index],
                    video[upper_index],
                    timestep=timestep,
                    model="4.25",
                    scale=scale,
                )

                progress_bar.update(1)

            output_frames.append(
                frame
            )

        return torch.stack(
            output_frames,
            dim=0,
        )


# ---------------------------------------------------------------------------
# Shorten Video
# ---------------------------------------------------------------------------

class ShortenVideo(IO.ComfyNode):
    """
    Shortens an IMAGE batch interpreted as a video sequence.

    If the input already contains at most n_frames, it is returned
    unchanged.
    """

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ShortenVideo_KASKI",
            display_name="Shorten Video",
            category="KASKI/videoTools",
            inputs=[
                IO.Image.Input(
                    "video",
                    tooltip=(
                        "Input image sequence to shorten if it contains more "
                        "than the requested number of frames."
                    ),
                ),
                IO.Int.Input(
                    "n_frames",
                    default=25,
                    min=1,
                    max=99999,
                    tooltip="Maximum number of frames the output sequence should contain.",
                ),
                IO.Combo.Input(
                    "method",
                    options=[
                        "cut_end",
                        "cut_beginning",
                        "resample",
                    ],
                    default="cut_end",
                    tooltip=(
                        "How frames are removed: trim the end, trim the beginning, "
                        "or evenly resample the full sequence."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="video",
                    tooltip="Input sequence shortened to at most the requested frame count.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video: torch.Tensor,
        n_frames: int,
        method: str,
    ) -> IO.NodeOutput:
        if video.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(video.shape)}"
            )

        current_frames = video.shape[0]

        if current_frames == 0:
            raise ValueError("Video contains no frames.")

        if current_frames <= n_frames:
            return IO.NodeOutput(video)

        if method == "cut_end":
            output = video[:n_frames]

        elif method == "cut_beginning":
            output = video[-n_frames:]

        elif method == "resample":
            output = cls._resample(
                video,
                n_frames,
            )

        else:
            raise ValueError(
                f"Unknown shortening method: {method}"
            )

        return IO.NodeOutput(
            output.contiguous()
        )

    @staticmethod
    def _resample(
        video: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        source_frames = video.shape[0]

        indices = torch.linspace(
            0,
            source_frames - 1,
            steps=target_frames,
            device=video.device,
        )

        indices = (
            indices
            .round()
            .long()
            .clamp(0, source_frames - 1)
        )

        return video[indices]


# ---------------------------------------------------------------------------
# Temporal Smoother
# ---------------------------------------------------------------------------

class TemporalSmoother(IO.ComfyNode):

    ANALYSIS_SCALE = 0.2
    TEMPORAL_RADIUS = 2

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="TemporalSmoother_KASKI",
            display_name="Temporal Smoother",
            category="KASKI/videoTools",
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip="Input IMAGE batch interpreted as a video sequence.",
                ),
                IO.Boolean.Input(
                    "resample",
                    default=True,
                    tooltip="Apply temporal resampling. Analysis always runs.",
                ),
                IO.Combo.Input(
                    "resample_method",
                    options=[
                        "rife",
                        "blend",
                    ],
                    default="rife",
                    tooltip="Method used to generate additional temporal samples.",
                ),
                IO.Float.Input(
                    "sensitivity",
                    default=0.7,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "Controls how strongly motion deviations affect temporal "
                        "resampling. 0 disables correction, 1 uses measured motion "
                        "directly, values above 1 increase correction strength."
                    ),
                ),
                # UI-Logik:
                # würde ich immer sichtbar lassen, weil Analyse immer läuft
                IO.Float.Input(
                    "scale_analysis",
                    default=1.0,
                    min=0.05,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "RIFE internal scale used for motion analysis optical flow. "
                        "Separate from the fixed image downscale used before analysis."
                    ),
                ),
                # UI-Logik:
                # nur anzeigen, wenn resample == True and resample_method == "rife"
                IO.Float.Input(
                    "scale_fill",
                    default=1.0,
                    min=0.05,
                    max=4.0,
                    step=0.05,
                    tooltip=(
                        "RIFE internal scale used when filling inserted frames. "
                        "Only relevant when resample is enabled and resample_method is rife."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(
                    display_name="images",
                ),
                IO.String.Output(
                    display_name="motion_table",
                ),
            ],
        )

    @classmethod
    def _downscale_for_analysis(
        cls,
        images: torch.Tensor,
    ) -> torch.Tensor:
        x = images.permute(0, 3, 1, 2).float()

        height, width = x.shape[2:4]

        target_height = max(
            1,
            round(height * cls.ANALYSIS_SCALE),
        )

        target_width = max(
            1,
            round(width * cls.ANALYSIS_SCALE),
        )

        x = F.interpolate(
            x,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _calculate_frame_motion(
        images: torch.Tensor,
        scale_analysis: float,
    ) -> torch.Tensor:
        motion_scores = []

        for i in range(images.shape[0] - 1):
            flow_1, flow_2 = calculate_optical_flow(
                images[i],
                images[i + 1],
                model="4.25",
                scale=scale_analysis,
            )

            magnitude_1 = torch.linalg.vector_norm(
                flow_1,
                dim=-1,
            ).mean()

            magnitude_2 = torch.linalg.vector_norm(
                flow_2,
                dim=-1,
            ).mean()

            motion_scores.append(
                (magnitude_1 + magnitude_2) * 0.5
            )

        return torch.stack(
            motion_scores
        )

    @classmethod
    def _calculate_motion_baseline(
        cls,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        count = motion.shape[0]
        baseline = torch.empty_like(motion)

        for i in range(count):
            start = max(
                0,
                i - cls.TEMPORAL_RADIUS,
            )

            end = min(
                count,
                i + cls.TEMPORAL_RADIUS + 1,
            )

            before = motion[start:i]
            after = motion[i + 1:end]

            if (
                before.numel() > 0
                and after.numel() > 0
            ):
                neighbors = torch.cat(
                    (before, after)
                )

            elif before.numel() > 0:
                neighbors = before

            elif after.numel() > 0:
                neighbors = after

            else:
                baseline[i] = motion[i]
                continue

            baseline[i] = torch.quantile(
                neighbors,
                0.5,
            )

        return baseline

    @staticmethod
    def _calculate_motion_ratio(
        motion: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        ratio = torch.ones_like(
            motion
        )

        valid = baseline > 1e-8

        ratio[valid] = (
            motion[valid]
            / baseline[valid]
        )

        return ratio.clamp_min(
            0.0
        )

    @staticmethod
    def _apply_sensitivity(
        motion_ratio: torch.Tensor,
        sensitivity: float,
    ) -> torch.Tensor:
        return (
            motion_ratio
            .clamp_min(1e-8)
            .pow(sensitivity)
        )

    @staticmethod
    def _calculate_target_intervals(
        adjusted_ratio: torch.Tensor,
    ) -> torch.Tensor:
        return torch.floor(
            adjusted_ratio + 0.5
        ).to(
            dtype=torch.int64
        ).clamp_min(
            0
        )

    @classmethod
    def _analyze_images(
        cls,
        images: torch.Tensor,
        sensitivity: float,
        scale_analysis: float,
    ):
        analysis_images = cls._downscale_for_analysis(
            images
        )

        motion = cls._calculate_frame_motion(
            analysis_images,
            scale_analysis,
        )

        baseline = cls._calculate_motion_baseline(
            motion
        )

        motion_ratio = cls._calculate_motion_ratio(
            motion,
            baseline,
        )

        adjusted_ratio = cls._apply_sensitivity(
            motion_ratio,
            sensitivity,
        )

        target_intervals = cls._calculate_target_intervals(
            adjusted_ratio
        )

        return (
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        )

    @staticmethod
    def _generate_table(
        motion: torch.Tensor,
        baseline: torch.Tensor,
        motion_ratio: torch.Tensor,
        adjusted_ratio: torch.Tensor,
        target_intervals: torch.Tensor,
    ) -> list:
        motion_cpu = (
            motion
            .detach()
            .float()
            .cpu()
        )

        baseline_cpu = (
            baseline
            .detach()
            .float()
            .cpu()
        )

        ratio_cpu = (
            motion_ratio
            .detach()
            .float()
            .cpu()
        )

        adjusted_cpu = (
            adjusted_ratio
            .detach()
            .float()
            .cpu()
        )

        intervals_cpu = (
            target_intervals
            .detach()
            .cpu()
        )

        table = [{
            "frame": 0,
            "motion": float("inf"),
            "baseline": float("inf"),
            "motion_ratio": 1.0,
            "adjusted_ratio": 1.0,
            "target_intervals": 0,
            "action": "START",
        }]

        for transition_index in range(
            motion_cpu.shape[0]
        ):
            frame_index = (
                transition_index + 1
            )

            intervals = int(
                intervals_cpu[
                    transition_index
                ].item()
            )

            if intervals == 0:
                action = "COLLAPSE"

            elif intervals == 1:
                action = "KEEP"

            else:
                action = (
                    f"INSERT {intervals - 1}"
                )

            table.append({
                "frame": frame_index,
                "motion": motion_cpu[
                    transition_index
                ].item(),
                "baseline": baseline_cpu[
                    transition_index
                ].item(),
                "motion_ratio": ratio_cpu[
                    transition_index
                ].item(),
                "adjusted_ratio": adjusted_cpu[
                    transition_index
                ].item(),
                "target_intervals": intervals,
                "action": action,
            })

        return table

    @staticmethod
    def _table_to_string(
        table: list,
    ) -> str:
        rows = [
            "Frame | Motion     | Baseline   | Ratio    | Adjusted | Intervals | Action",
            "------+------------+------------+----------+----------+-----------+---------",
        ]

        for entry in table:
            frame_index = entry["frame"]

            if frame_index == 0:
                motion = "inf"
                baseline = "inf"
                ratio = "1.000000"
                adjusted = "1.000000"
                intervals = "-"

            else:
                motion = (
                    f"{entry['motion']:.6f}"
                )

                baseline = (
                    f"{entry['baseline']:.6f}"
                )

                ratio = (
                    f"{entry['motion_ratio']:.6f}"
                )

                adjusted = (
                    f"{entry['adjusted_ratio']:.6f}"
                )

                intervals = str(
                    entry["target_intervals"]
                )

            rows.append(
                f"{frame_index:<5} | "
                f"{motion:<10} | "
                f"{baseline:<10} | "
                f"{ratio:<8} | "
                f"{adjusted:<8} | "
                f"{intervals:<9} | "
                f"{entry['action']}"
            )

        return "\n".join(
            rows
        )

    @staticmethod
    def _resample_rife(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
        scale_fill: float,
    ) -> torch.Tensor:
        return interpolate_between_two_frames(
            image_1,
            image_2,
            timestep=timestep,
            model="4.25",
            scale=scale_fill,
        )

    @staticmethod
    def _resample_blend(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:
        return torch.lerp(
            image_1,
            image_2,
            timestep,
        )

    @classmethod
    def _resample_frame(
        cls,
        resample_method: str,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
        scale_fill: float,
    ) -> torch.Tensor:
        if resample_method == "rife":
            return cls._resample_rife(
                image_1,
                image_2,
                timestep,
                scale_fill,
            )

        if resample_method == "blend":
            return cls._resample_blend(
                image_1,
                image_2,
                timestep,
            )

        raise ValueError(
            f"Unknown resample method: {resample_method}"
        )

    @classmethod
    def _resample_motion(
        cls,
        images: torch.Tensor,
        table: list,
        resample_method: str,
        scale_fill: float,
    ) -> torch.Tensor:
        output_frames = [
            images[0]
        ]

        generated_frames = sum(
            max(
                entry["target_intervals"] - 1,
                0,
            )
            for entry in table[1:]
        )

        progress_bar = ProgressBar(
            generated_frames
        )

        for transition_index, entry in enumerate(
            table[1:]
        ):
            image_1 = images[
                transition_index
            ]

            image_2 = images[
                transition_index + 1
            ]

            interval_count = entry[
                "target_intervals"
            ]

            if interval_count == 0:
                output_frames[-1] = image_2
                continue

            for step in range(
                1,
                interval_count,
            ):
                timestep = (
                    step
                    / interval_count
                )

                output_frames.append(
                    cls._resample_frame(
                        resample_method,
                        image_1,
                        image_2,
                        timestep,
                        scale_fill,
                    )
                )

                progress_bar.update(1)

            output_frames.append(
                image_2
            )

        return torch.stack(
            output_frames,
            dim=0,
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        resample: bool,
        resample_method: str,
        sensitivity: float,
        scale_analysis: float,
        scale_fill: float,
    ) -> IO.NodeOutput:
        if images.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), "
                f"got {tuple(images.shape)}"
            )

        frame_count = images.shape[0]

        if frame_count == 0:
            return IO.NodeOutput(
                images,
                (
                    "Frame | Motion     | Baseline   | Ratio    | Adjusted | Intervals | Action\n"
                    "------+------------+------------+----------+----------+-----------+---------"
                ),
            )

        if frame_count == 1:
            table = [{
                "frame": 0,
                "motion": float("inf"),
                "baseline": float("inf"),
                "motion_ratio": 1.0,
                "adjusted_ratio": 1.0,
                "target_intervals": 0,
                "action": "START",
            }]

            return IO.NodeOutput(
                images,
                cls._table_to_string(
                    table
                ),
            )

        (
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        ) = cls._analyze_images(
            images,
            sensitivity,
            scale_analysis,
        )

        table = cls._generate_table(
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        )

        motion_table = cls._table_to_string(
            table
        )

        if resample:
            output_images = cls._resample_motion(
                images,
                table,
                resample_method,
                scale_fill,
            )
        else:
            output_images = images

        return IO.NodeOutput(
            output_images.contiguous(),
            motion_table,
        )


# ---------------------------------------------------------------------------
# V3 registration
# ---------------------------------------------------------------------------

VIDEO_TOOLS_NODES_LIST = [
    ExtendVideo,
    ShortenVideo,
    TemporalSmoother,
]