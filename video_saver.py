import base64
import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import IO
from comfy.cli_args import args
from comfy.utils import ProgressBar


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Codec configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProResProfile:
    name: str
    ffmpeg_profile: int
    chroma: str
    target_bit_depth: int
    supports_alpha: bool


PRORES_PROFILES: dict[str, ProResProfile] = {
    "ProRes 422 Proxy": ProResProfile(
        name="ProRes 422 Proxy",
        ffmpeg_profile=0,
        chroma="422",
        target_bit_depth=10,
        supports_alpha=False,
    ),
    "ProRes 422 LT": ProResProfile(
        name="ProRes 422 LT",
        ffmpeg_profile=1,
        chroma="422",
        target_bit_depth=10,
        supports_alpha=False,
    ),
    "ProRes 422": ProResProfile(
        name="ProRes 422",
        ffmpeg_profile=2,
        chroma="422",
        target_bit_depth=10,
        supports_alpha=False,
    ),
    "ProRes 422 HQ": ProResProfile(
        name="ProRes 422 HQ",
        ffmpeg_profile=3,
        chroma="422",
        target_bit_depth=10,
        supports_alpha=False,
    ),
    "ProRes 4444": ProResProfile(
        name="ProRes 4444",
        ffmpeg_profile=4,
        chroma="444",
        target_bit_depth=10,
        supports_alpha=True,
    ),
    "ProRes 4444 XQ": ProResProfile(
        name="ProRes 4444 XQ",
        ffmpeg_profile=5,
        chroma="444",
        target_bit_depth=12,
        supports_alpha=True,
    ),
}


def get_profile(name: str) -> ProResProfile:
    try:
        return PRORES_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown ProRes profile: {name}") from exc


# ---------------------------------------------------------------------------
# FFmpeg discovery / capability probing
# ---------------------------------------------------------------------------


def resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    except (ImportError, RuntimeError):
        pass

    raise RuntimeError(
        "FFmpeg could not be found. "
        "Install imageio-ffmpeg or make ffmpeg available on PATH."
    )


@lru_cache(maxsize=8)
def probe_prores_ks_pixel_formats(ffmpeg_path: str) -> frozenset[str]:
    """Return pixel formats advertised by the installed prores_ks encoder."""
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-h",
                "encoder=prores_ks",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not probe FFmpeg ProRes support: {exc}") from exc

    text = result.stdout or ""
    if result.returncode != 0 or "prores_ks" not in text:
        raise RuntimeError(
            "The resolved FFmpeg executable does not provide the prores_ks encoder."
        )

    formats: set[str] = set()
    marker = "Supported pixel formats:"

    for line in text.splitlines():
        if marker in line:
            formats.update(line.split(marker, 1)[1].strip().split())

    return frozenset(formats)


def choose_output_pixel_format(
    profile: ProResProfile,
    use_alpha: bool,
    ffmpeg_path: str,
) -> str:
    formats = probe_prores_ks_pixel_formats(ffmpeg_path)

    if profile.target_bit_depth == 10:
        if profile.chroma == "422":
            candidate = "yuv422p10le"
        else:
            candidate = "yuva444p10le" if use_alpha else "yuv444p10le"

    elif profile.target_bit_depth == 12:
        candidate = "yuva444p12le" if use_alpha else "yuv444p12le"

    else:
        raise RuntimeError(
            f"Unsupported configured ProRes bit depth: {profile.target_bit_depth}"
        )

    # FFmpeg often reports native aliases without endian suffixes on LE hosts.
    accepted_names = {
        candidate,
        candidate.removesuffix("le"),
    }

    if not formats.intersection(accepted_names):
        if profile.target_bit_depth == 12:
            raise RuntimeError(
                "ProRes 4444 XQ is configured as a strict 12-bit output, but this "
                "FFmpeg prores_ks build does not advertise a 12-bit 4:4:4 input "
                "pixel format. KASKI will not silently downgrade XQ to 10-bit."
            )

        raise RuntimeError(
            f"FFmpeg prores_ks does not advertise the required pixel format "
            f"{candidate} for {profile.name}. Advertised formats: "
            f"{', '.join(sorted(formats)) or 'unknown'}"
        )

    return candidate


# ---------------------------------------------------------------------------
# Input validation / raw frame conversion
# ---------------------------------------------------------------------------


def validate_images(images: torch.Tensor) -> tuple[int, int, int]:
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a torch.Tensor.")

    if images.ndim != 4:
        raise ValueError(
            f"Expected IMAGE tensor (B,H,W,C), got {tuple(images.shape)}."
        )

    frame_count, height, width, channels = images.shape

    if frame_count < 1 or height < 1 or width < 1:
        raise ValueError("IMAGE tensor must not be empty.")

    if channels != 3:
        raise ValueError(
            f"KASKI Save ProRes expects RGB IMAGE tensors with 3 channels, got {channels}."
        )

    # Both FFmpeg ProRes encoders require an even frame width.
    if width % 2 != 0:
        raise ValueError(
            f"ProRes requires an even frame width. Received {width}x{height}."
        )

    return frame_count, height, width


def validate_mask(mask: torch.Tensor | None, images: torch.Tensor) -> None:
    if mask is None:
        return

    if not isinstance(mask, torch.Tensor):
        raise TypeError("mask must be a torch.Tensor.")

    if mask.ndim != 3:
        raise ValueError(
            f"Expected MASK tensor (B,H,W), got {tuple(mask.shape)}."
        )

    image_frames, height, width, _ = images.shape
    mask_frames, mask_h, mask_w = mask.shape

    if (mask_h, mask_w) != (height, width):
        raise ValueError(
            "MASK dimensions must match IMAGE dimensions exactly. "
            f"IMAGE is {width}x{height}; MASK is {mask_w}x{mask_h}."
        )

    if mask_frames not in (1, image_frames):
        raise ValueError(
            "MASK batch must contain either one frame (broadcast to the whole video) "
            f"or exactly {image_frames} frames; got {mask_frames}."
        )


def _frame_to_u16le(frame: torch.Tensor) -> bytes:
    array = (
        frame.detach()
        .to(dtype=torch.float32)
        .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0.0, 1.0)
        .mul(65535.0)
        .round()
        .to(torch.int32)
        .cpu()
        .numpy()
        .astype("<u2", copy=False)
    )
    return np.ascontiguousarray(array).tobytes()


def iter_raw_frames(
    images: torch.Tensor,
    mask: torch.Tensor | None,
    use_alpha: bool,
):
    frame_count = images.shape[0]

    for index in range(frame_count):
        rgb = images[index]

        if use_alpha:
            assert mask is not None
            alpha = mask[0 if mask.shape[0] == 1 else index]
            rgba = torch.cat((rgb, alpha.unsqueeze(-1)), dim=-1)
            yield _frame_to_u16le(rgba)
        else:
            yield _frame_to_u16le(rgb)


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------


def validate_audio(audio) -> tuple[torch.Tensor, int] | None:
    if audio is None:
        return None

    if not isinstance(audio, dict):
        raise TypeError("AUDIO input must be a ComfyUI AUDIO dictionary.")

    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("AUDIO input must contain waveform and sample_rate.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])

    if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
        raise ValueError(
            "AUDIO waveform must be a tensor with shape (B,C,T)."
        )

    if waveform.shape[0] != 1:
        raise ValueError(
            f"KASKI Save ProRes supports one AUDIO item at a time; got batch {waveform.shape[0]}."
        )

    if waveform.shape[1] < 1:
        raise ValueError("AUDIO must contain at least one channel.")

    if sample_rate <= 0:
        raise ValueError(f"Invalid AUDIO sample rate: {sample_rate}")

    if waveform.shape[-1] == 0:
        return None

    return waveform, sample_rate


def get_audio_duration(waveform: torch.Tensor, sample_rate: int) -> float:
    return waveform.shape[-1] / float(sample_rate)


def build_atempo_chain(speed_factor: float) -> list[str]:
    """Split tempo changes into conservative FFmpeg atempo stages [0.5, 2.0]."""
    if speed_factor <= 0 or not np.isfinite(speed_factor):
        raise ValueError(f"Invalid audio tempo factor: {speed_factor}")

    filters: list[str] = []
    remaining = float(speed_factor)

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    if not np.isclose(remaining, 1.0, rtol=0.0, atol=1e-9):
        filters.append(f"atempo={remaining:.12g}")

    return filters


def build_audio_filter(
    source_duration: float,
    target_duration: float,
) -> str:
    if target_duration <= 0:
        raise ValueError("Target video duration must be greater than zero.")

    filters: list[str] = []
    duration_delta = abs(source_duration - target_duration)

    # Avoid an unnecessary time stretch for tiny timestamp/sample rounding noise.
    if duration_delta > 0.0005:
        speed_factor = source_duration / target_duration
        filters.extend(build_atempo_chain(speed_factor))

    # apad handles sub-sample rounding that would otherwise leave the audio a few
    # samples short; atrim guarantees the final stream ends exactly with the video.
    filters.append("apad")
    filters.append(f"atrim=duration={target_duration:.12g}")

    return ",".join(filters)


def write_temp_audio_raw(waveform: torch.Tensor) -> Path:
    fd, raw_path = tempfile.mkstemp(prefix="kaski_prores_audio_", suffix=".f32le")
    os.close(fd)
    path = Path(raw_path)

    # Comfy AUDIO: (1, channels, samples). FFmpeg f32le expects interleaved samples.
    interleaved = (
        waveform[0]
        .detach()
        .to(dtype=torch.float32)
        .nan_to_num(nan=0.0, posinf=1.0, neginf=-1.0)
        .clamp(-1.0, 1.0)
        .transpose(0, 1)
        .contiguous()
        .cpu()
        .numpy()
        .astype("<f4", copy=False)
    )

    path.write_bytes(interleaved.tobytes())
    return path


# ---------------------------------------------------------------------------
# Poster frame
# ---------------------------------------------------------------------------


def encode_poster_frame_jpeg(
    images: torch.Tensor,
    quality: int = 90,
) -> str:
    """
    Encode frame 0 as an 8-bit JPEG and return it as Base64 text.

    This is only a documentation preview for tools that cannot decode ProRes
    natively. It does not affect the encoded video stream.
    """
    frame = (
        images[0]
        .detach()
        .to(dtype=torch.float32)
        .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(
            f"Poster frame expects RGB data, got shape {tuple(frame.shape)}."
        )

    buffer = io.BytesIO()
    Image.fromarray(frame, mode="RGB").save(
        buffer,
        format="JPEG",
        quality=int(quality),
        optimize=True,
    )

    return base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def normalize_reference_files(reference_files) -> list[str]:
    if not reference_files:
        return []

    references: list[str] = []

    for value in reference_files.values():
        if value is None:
            continue
        text = str(value).strip()
        if text:
            references.append(text)

    return references


def build_mov_metadata(
    model_name: str,
    user_prompt: str,
    seed,
    reference_files: list[str],
    poster_frame_jpeg: str,
    workflow_prompt,
    extra_pnginfo,
) -> dict[str, str]:
    metadata: dict[str, str] = {}

    if args.disable_metadata:
        return metadata

    if workflow_prompt is not None:
        metadata["prompt"] = json.dumps(
            workflow_prompt,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            metadata.setdefault(
                str(key),
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )

    if model_name:
        metadata["model_name"] = str(model_name)

    if user_prompt:
        metadata["user_prompt"] = str(user_prompt)

    seed_text = str(seed).strip() if seed is not None else ""
    if seed_text and seed_text != "-1":
        metadata["seed"] = seed_text

    if reference_files:
        metadata["reference_files"] = json.dumps(
            reference_files,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if poster_frame_jpeg:
        metadata["poster_frame_jpeg"] = poster_frame_jpeg

    return metadata


def _escape_ffmetadata(value: str) -> str:
    out: list[str] = []

    for char in value.replace("\r\n", "\n").replace("\r", "\n"):
        if char == "\n":
            out.append("\\\n")
        elif char in "#;=\\":
            out.append("\\" + char)
        elif char == "\x00":
            continue
        else:
            out.append(char)

    return "".join(out)


def write_temp_ffmetadata(metadata: dict[str, str]) -> Path | None:
    if not metadata:
        return None

    fd, metadata_path = tempfile.mkstemp(
        prefix="kaski_prores_metadata_",
        suffix=".ffmeta",
    )
    os.close(fd)
    path = Path(metadata_path)

    lines = [";FFMETADATA1"]

    for key, value in metadata.items():
        lines.append(
            f"{_escape_ffmetadata(str(key))}={_escape_ffmetadata(str(value))}"
        )

    # IMPORTANT:
    # FFmetadata multiline escaping requires literal LF line endings.
    # Do not let Windows convert them to CRLF.
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write("\n".join(lines) + "\n")

    return path


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------


def resolve_output_path(
    filename_prefix: str,
    images: torch.Tensor,
) -> tuple[Path, str, str]:
    output_dir = folder_paths.get_output_directory()

    full_output_folder, filename, counter, subfolder, _ = (
        folder_paths.get_save_image_path(
            filename_prefix,
            output_dir,
            images.shape[2],
            images.shape[1],
        )
    )

    os.makedirs(full_output_folder, exist_ok=True)

    while True:
        file = f"{filename}_{counter:05}_.mov"
        path = Path(full_output_folder) / file
        if not path.exists():
            return path, file, subfolder
        counter += 1


# ---------------------------------------------------------------------------
# Encoder lifecycle
# ---------------------------------------------------------------------------


class ProResEncoder:
    def __init__(
        self,
        ffmpeg_path: str,
        profile: ProResProfile,
        framerate: float,
        output_path: Path,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.profile = profile
        self.framerate = float(framerate)
        self.output_path = Path(output_path)

    def encode(
        self,
        images: torch.Tensor,
        mask: torch.Tensor | None = None,
        audio=None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        frame_count, height, width = validate_images(images)
        validate_mask(mask, images)

        if self.framerate <= 0 or not np.isfinite(self.framerate):
            raise ValueError(f"Invalid framerate: {self.framerate}")

        if mask is not None and not self.profile.supports_alpha:
            log.warning(
                "[KASKI Save ProRes] MASK is connected, but %s does not support alpha. "
                "The MASK will be ignored.",
                self.profile.name,
            )

        use_alpha = mask is not None and self.profile.supports_alpha
        output_pix_fmt = choose_output_pixel_format(
            self.profile,
            use_alpha,
            self.ffmpeg_path,
        )
        input_pix_fmt = "rgba64le" if use_alpha else "rgb48le"
        target_duration = frame_count / self.framerate

        validated_audio = validate_audio(audio)
        audio_path: Path | None = None
        metadata_path: Path | None = None

        try:
            audio_info = None
            if validated_audio is not None:
                waveform, sample_rate = validated_audio
                audio_path = write_temp_audio_raw(waveform)
                audio_info = {
                    "sample_rate": sample_rate,
                    "channels": int(waveform.shape[1]),
                    "filter": build_audio_filter(
                        get_audio_duration(waveform, sample_rate),
                        target_duration,
                    ),
                }

            metadata_path = write_temp_ffmetadata(metadata or {})

            command = self._build_command(
                width=width,
                height=height,
                input_pix_fmt=input_pix_fmt,
                output_pix_fmt=output_pix_fmt,
                audio_path=audio_path,
                audio_info=audio_info,
                metadata_path=metadata_path,
            )

            self._run_ffmpeg(
                command=command,
                images=images,
                mask=mask,
                use_alpha=use_alpha,
                frame_count=frame_count,
            )

        except Exception:
            # Never leave a corrupt partial MOV behind after a failed encode.
            try:
                if self.output_path.exists():
                    self.output_path.unlink()
            except OSError:
                pass
            raise

        finally:
            for temp_path in (audio_path, metadata_path):
                if temp_path is None:
                    continue
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _build_command(
        self,
        width: int,
        height: int,
        input_pix_fmt: str,
        output_pix_fmt: str,
        audio_path: Path | None,
        audio_info: dict | None,
        metadata_path: Path | None,
    ) -> list[str]:
        command = [
            self.ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            input_pix_fmt,
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{self.framerate:.12g}",
            "-i",
            "pipe:0",
        ]

        audio_input_index = None
        metadata_input_index = None
        next_input_index = 1

        if audio_path is not None:
            assert audio_info is not None
            audio_input_index = next_input_index
            next_input_index += 1
            command += [
                "-f",
                "f32le",
                "-ar",
                str(audio_info["sample_rate"]),
                "-ac",
                str(audio_info["channels"]),
                "-i",
                str(audio_path),
            ]

        if metadata_path is not None:
            metadata_input_index = next_input_index
            command += [
                "-f",
                "ffmetadata",
                "-i",
                str(metadata_path),
            ]

        command += [
            "-map",
            "0:v:0",
        ]

        if audio_input_index is not None:
            command += [
                "-map",
                f"{audio_input_index}:a:0",
            ]

        if metadata_input_index is not None:
            command += [
                "-map_metadata",
                str(metadata_input_index),
            ]
        else:
            command += ["-map_metadata", "-1"]

        command += [
            "-c:v",
            "prores_ks",
            "-profile:v",
            str(self.profile.ffmpeg_profile),
            "-pix_fmt",
            output_pix_fmt,
        ]

        if audio_input_index is not None:
            command += [
                "-af",
                str(audio_info["filter"]),
                "-c:a",
                "pcm_s24le",
            ]

        if metadata_input_index is not None:
            command += [
                "-movflags",
                "use_metadata_tags",
            ]

        command.append(str(self.output_path))
        return command

    def _run_ffmpeg(
        self,
        command: list[str],
        images: torch.Tensor,
        mask: torch.Tensor | None,
        use_alpha: bool,
        frame_count: int,
    ) -> None:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        assert process.stdin is not None
        progress = ProgressBar(frame_count)

        try:
            for frame_bytes in iter_raw_frames(images, mask, use_alpha):
                process.stdin.write(frame_bytes)
                progress.update(1)

            process.stdin.close()
            process.stdin = None
            _, stderr = process.communicate()

        except Exception:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except Exception:
                pass
            process.kill()
            process.wait()
            raise

        if process.returncode != 0:
            error_text = (stderr or b"").decode(
                "utf-8",
                errors="replace",
            ).strip()
            raise RuntimeError(
                "FFmpeg ProRes encode failed"
                + (f":\n{error_text}" if error_text else ".")
            )


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------


class SaveProRes(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_SaveProRes",
            display_name="Save Video as ProRes",
            description=(
                "Saves an IMAGE batch as a ProRes MOV. Video is fed to FFmpeg as "
                "16-bit RGB(A), encoded to at least 10-bit ProRes, and optional "
                "audio is pitch-preservingly retimed to the output video duration."
            ),
            category="KASKI/savers",
            search_aliases=[
                "save prores",
                "save mov",
                "prores",
                "export video",
            ],
            inputs=[
                IO.Image.Input(
                    "images",
                    tooltip="IMAGE batch interpreted as consecutive video frames.",
                ),
                IO.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Optional AUDIO. Its duration is pitch-preservingly adjusted "
                        "to exactly frame_count / framerate."
                    ),
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional alpha channel. White = visible, black = transparent. "
                        "Used only by ProRes profiles that support alpha."
                    ),
                ),
                IO.Combo.Input(
                    "prores_type",
                    options=list(PRORES_PROFILES.keys()),
                    default="ProRes 422 HQ",
                    tooltip=(
                        "422 profiles are 10-bit. 4444 is 10-bit with optional alpha. "
                        "4444 XQ is strict 12-bit and will error instead of silently "
                        "falling back if the installed FFmpeg encoder cannot provide it."
                    ),
                ),
                IO.Float.Input(
                    "framerate",
                    default=25.0,
                    min=0.01,
                    max=1000.0,
                    step=0.001,
                ),
                IO.String.Input(
                    "filename_prefix",
                    default="video/ComfyUI",
                    tooltip="Output filename prefix. Supports ComfyUI formatting tokens.",
                ),
                IO.String.Input(
                    "model_name",
                    default="",
                    optional=True,
                    tooltip="Optional model name stored in MOV metadata.",
                ),
                IO.String.Input(
                    "user_prompt",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip="Optional generation prompt stored in MOV metadata.",
                ),
                IO.Int.Input(
                    "seed",
                    default=-1,
                    optional=True,
                    tooltip="Optional generation seed stored in MOV metadata. -1 omits the field.",
                ),
                IO.Autogrow.Input(
                    "reference_files",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.String.Input(
                            "reference",
                            force_input=True,
                        ),
                        prefix="reference_",
                        min=0,
                        max=50,
                    ),
                    tooltip=(
                        "Optional reference filenames. Connected values are stored in "
                        "MOV metadata as a JSON array in input order."
                    ),
                ),
            ],
            hidden=[
                IO.Hidden.prompt,
                IO.Hidden.extra_pnginfo,
            ],
            is_output_node=True,
            outputs=[
                IO.Image.Output(display_name="images"),
                IO.String.Output(display_name="filepath"),
            ],
        )

    @classmethod
    def execute(
        cls,
        images: torch.Tensor,
        audio=None,
        mask: torch.Tensor | None = None,
        prores_type: str = "ProRes 422 HQ",
        framerate: float = 25.0,
        filename_prefix: str = "video/ComfyUI",
        model_name: str = "",
        user_prompt: str = "",
        seed: int = -1,
        reference_files: IO.Autogrow.Type | None = None,
    ) -> IO.NodeOutput:
        profile = get_profile(prores_type)
        ffmpeg_path = resolve_ffmpeg()

        references = normalize_reference_files(reference_files)
        poster_frame_jpeg = (
            ""
            if args.disable_metadata
            else encode_poster_frame_jpeg(images)
        )

        metadata = build_mov_metadata(
            model_name=model_name,
            user_prompt=user_prompt,
            seed=seed,
            reference_files=references,
            poster_frame_jpeg=poster_frame_jpeg,
            workflow_prompt=cls.hidden.prompt,
            extra_pnginfo=cls.hidden.extra_pnginfo,
        )

        output_path, file, subfolder = resolve_output_path(
            filename_prefix,
            images,
        )

        encoder = ProResEncoder(
            ffmpeg_path=ffmpeg_path,
            profile=profile,
            framerate=framerate,
            output_path=output_path,
        )

        encoder.encode(
            images=images,
            mask=mask,
            audio=audio,
            metadata=metadata,
        )

        return IO.NodeOutput(
            images,
            str(output_path),
            ui={
                "images": [
                    {
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ],
                "animated": (True,),
            },
        )


PRORES_SAVER_NODE_LIST = [
    SaveProRes,
]
