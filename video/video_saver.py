import base64
import io
import json
import logging
import mmap
import os
import shutil
import struct
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


SDR_TRANSFER_TAGS = {"sRGB": "iec61966-2-1", "Rec.709": "bt709"}


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

    def collect(value):
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)
        elif value is not None:
            text = str(value).strip()
            if text:
                references.append(text)

    collect(reference_files)

    return references


def build_mov_metadata(
    model_name: str,
    user_prompt: str,
    seed,
    reference_files: list[str],
    poster_frame_jpeg: str,
    workflow_prompt,
    extra_pnginfo,
    save_workflow: bool = True,
) -> dict[str, str]:
    metadata: dict[str, str] = {}

    if args.disable_metadata:
        return metadata

    if save_workflow and workflow_prompt is not None:
        metadata["prompt"] = json.dumps(
            workflow_prompt,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    if extra_pnginfo:
        for key, value in extra_pnginfo.items():
            if not save_workflow and key in ("prompt", "workflow"):
                continue
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


def resolve_output_location(
    filename_prefix: str,
    images: torch.Tensor,
) -> tuple[Path, str, int, str]:
    output_dir = folder_paths.get_output_directory()
    full_output_folder, filename, counter, subfolder, _ = (
        folder_paths.get_save_image_path(
            filename_prefix,
            output_dir,
            images.shape[2],
            images.shape[1],
        )
    )
    output_folder = Path(full_output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    return output_folder, filename, counter, subfolder


def publish_output(
    encoded_path: Path,
    output_folder: Path,
    filename: str,
    counter: int,
) -> Path:
    while True:
        output_path = output_folder / f"{filename}_{counter:05}_.mov"
        try:
            # Windows rename fails when the destination exists. On POSIX,
            # linking gives us the same atomic, no-overwrite behavior.
            if os.name == "nt":
                encoded_path.rename(output_path)
            else:
                os.link(encoded_path, output_path)
            return output_path
        except FileExistsError:
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
        input_transfer: str = "sRGB",
    ):
        self.ffmpeg_path = ffmpeg_path
        self.profile = profile
        self.framerate = float(framerate)
        self.output_path = Path(output_path)
        self.transfer_tag = SDR_TRANSFER_TAGS[input_transfer]

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
            "-vf",
            "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,"
            f"setparams=range=limited:color_primaries=bt709:color_trc={self.transfer_tag}"
            ":colorspace=bt709",
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
# MOV descriptive metadata scrubber
# ---------------------------------------------------------------------------

_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}
_DESCRIPTION_ATOMS = {b"udta", b"meta", b"uuid"}


def _atoms(data, start, end):
    """Yield bounded QuickTime atoms; fail instead of silently skipping malformed data."""
    offset = start
    while offset < end:
        if end - offset < 8:
            raise ValueError("Truncated MOV atom header")
        size = struct.unpack_from(">I", data, offset)[0]
        kind = bytes(data[offset + 4:offset + 8])
        header = 8
        if size == 1:
            if end - offset < 16:
                raise ValueError("Truncated extended MOV atom header")
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header = 16
        elif size == 0:
            size = end - offset
        if size < header or size > end - offset:
            raise ValueError("Invalid MOV atom size")
        yield offset, offset + size, offset + header, kind
        offset += size


def _erase(data, start, end):
    # Keep atom sizes and sample offsets intact; turn removed atoms into free space.
    data[start + 4:start + 8] = b"free"
    header = 16 if struct.unpack_from(">I", data, start)[0] == 1 else 8
    for position in range(start + header, end, 1024 * 1024):
        data[position:min(position + 1024 * 1024, end)] = bytes(
            min(1024 * 1024, end - position)
        )


def _clear_header_dates(data, payload, end):
    if end - payload < 12:
        raise ValueError("Truncated MOV time header")
    width = 8 if data[payload] == 1 else 4
    if end - payload < 4 + 2 * width:
        raise ValueError("Truncated MOV time fields")
    data[payload + 4:payload + 4 + 2 * width] = bytes(2 * width)


def _clear_handler_name(data, payload, end):
    # The 24-byte hdlr preamble carries the functional handler type.
    if end - payload < 24:
        raise ValueError("Truncated MOV handler")
    data[payload + 24:end] = bytes(end - payload - 24)


def _clear_sample_descriptions(data, payload, end):
    if end - payload < 8:
        raise ValueError("Truncated MOV sample description table")
    count = struct.unpack_from(">I", data, payload + 4)[0]
    entries = list(_atoms(data, payload + 8, end))
    if len(entries) != count:
        raise ValueError("MOV sample description count does not match")
    for start, finish, _, codec in entries:
        # Sample entry type is the codec FOURCC; retain it, dimensions, and codec config.
        if codec in (b"apch", b"apcn", b"apcs", b"apco", b"ap4h", b"ap4x"):
            if finish - start < 86:
                raise ValueError("Truncated ProRes sample description")
            data[start + 20:start + 24] = bytes(4)  # vendor
            data[start + 50:start + 82] = bytes(32)  # compressor name
        elif codec in (b"in24", b"lpcm", b"sowt", b"twos"):
            if finish - start < 24:
                raise ValueError("Truncated PCM sample description")
            data[start + 20:start + 24] = bytes(4)  # vendor


def strip_mov_descriptive_metadata(path: Path):
    """Scrub QuickTime descriptive atoms and timestamps without re-encoding.

    Operates in place on a private temporary file, before it is published.
    Codec, color, dimensions, timebase and sample tables remain intact.
    """
    with open(path, "r+b") as output:
        with mmap.mmap(output.fileno(), 0, access=mmap.ACCESS_WRITE) as data:
            def walk(start, end):
                for atom_start, atom_end, payload, kind in _atoms(data, start, end):
                    if kind in _DESCRIPTION_ATOMS:
                        _erase(data, atom_start, atom_end)
                    elif kind in _CONTAINERS:
                        walk(payload, atom_end)
                    elif kind in (b"mvhd", b"tkhd", b"mdhd"):
                        _clear_header_dates(data, payload, atom_end)
                    elif kind == b"hdlr":
                        _clear_handler_name(data, payload, atom_end)
                    elif kind == b"stsd":
                        _clear_sample_descriptions(data, payload, atom_end)

            walk(0, len(data))
            data.flush()


class _CleanProResEncoder(ProResEncoder):
    def _build_command(self, *args, **kwargs):
        command = super()._build_command(*args, **kwargs)
        # Explicitly suppress input/automatic metadata before scrubbing muxer atoms.
        command[-1:-1] = [
            "-map_chapters", "-1",
            "-metadata", "encoder=",
            "-metadata:s:v:0", "encoder=",
            "-metadata:s:a:0", "encoder=",
        ]
        return command


def _list_value(values, index, default=None):
    if values is None:
        return default
    if not isinstance(values, list):
        return values
    return values[min(index, len(values) - 1)] if values else default


class _SaveProRes(IO.ComfyNode):
    """Each IMAGE list item is one (B,H,W,C) tensor, hence one MOV."""

    WITH_METADATA = False

    @classmethod
    def _inputs(cls):
        inputs = [
            IO.Image.Input("images", tooltip="List item: one (B,H,W,C) video batch; B is the frame count."),
            IO.Audio.Input("audio", optional=True, tooltip="Optional AUDIO per video; shorter lists repeat their final item."),
            IO.Mask.Input("mask", optional=True, tooltip="Optional alpha mask for 4444 profiles, per video."),
            IO.Combo.Input(
                "prores_type", options=list(PRORES_PROFILES.keys()),
                default="ProRes 422 HQ",
                tooltip="422 is 10-bit; 4444 supports alpha; XQ requires strict 12-bit encoder support.",
            ),
            IO.Float.Input("framerate", default=25.0, min=0.01, max=1000.0, step=0.001),
            IO.String.Input("filename_prefix", default="video/ComfyUI"),
        ]
        if cls.WITH_METADATA:
            inputs += [
                IO.String.Input("model_name", default="", optional=True),
                IO.String.Input("user_prompt", default="", multiline=True, optional=True),
                IO.Int.Input("seed", default=-1, optional=True),
                IO.Autogrow.Input(
                    "reference_files",
                    template=IO.Autogrow.TemplatePrefix(
                        input=IO.String.Input("reference", force_input=True),
                        prefix="reference_", min=0, max=50,
                    ),
                    tooltip="All reference filenames are metadata for each video; a reference list creates no additional MOV.",
                ),
                IO.Boolean.Input(
                    "save_workflow", default=True, display_name="Save Workflow",
                    tooltip="Embed ComfyUI execution prompt and workflow. Other custom metadata remains enabled.",
                ),
            ]
        inputs.append(
            IO.Combo.Input(
                "input_transfer", display_name="input transfer",
                options=list(SDR_TRANSFER_TAGS), default="sRGB", optional=True,
                tooltip="Transfer already used by the input RGB: sRGB or Rec.709.",
            )
        )
        return inputs

    @classmethod
    def execute(
        cls, images, audio=None, mask=None, prores_type=None, framerate=None,
        filename_prefix=None, model_name=None, user_prompt=None, seed=None,
        reference_files=None, input_transfer=None, save_workflow=None,
    ) -> IO.NodeOutput:
        if not isinstance(images, list):
            images = [images]
        if not images:
            raise ValueError("No IMAGE batches were provided.")

        references = normalize_reference_files(reference_files) if cls.WITH_METADATA else []
        paths = []
        previews = []
        ffmpeg_path = resolve_ffmpeg()
        for index, batch in enumerate(images):
            if not isinstance(batch, torch.Tensor) or batch.ndim != 4:
                raise ValueError("Each images list element must be an IMAGE tensor (B,H,W,C).")

            profile = get_profile(_list_value(prores_type, index, "ProRes 422 HQ"))
            fps = _list_value(framerate, index, 25.0)
            prefix = _list_value(filename_prefix, index, "video/ComfyUI")
            transfer = _list_value(input_transfer, index, "sRGB")
            video_audio = _list_value(audio, index)
            video_mask = _list_value(mask, index)

            metadata = {}
            if cls.WITH_METADATA and not args.disable_metadata:
                metadata = build_mov_metadata(
                    model_name=_list_value(model_name, index, ""),
                    user_prompt=_list_value(user_prompt, index, ""),
                    seed=_list_value(seed, index, -1),
                    reference_files=references,
                    poster_frame_jpeg=encode_poster_frame_jpeg(batch),
                    workflow_prompt=cls.hidden.prompt,
                    extra_pnginfo=cls.hidden.extra_pnginfo,
                    save_workflow=bool(_list_value(save_workflow, index, True)),
                )

            folder, name, counter, subfolder = resolve_output_location(prefix, batch)
            with tempfile.TemporaryDirectory(prefix=".kaski_prores_", dir=folder) as temp_dir:
                encoded_path = Path(temp_dir) / "video.mov"
                encoder_type = ProResEncoder if cls.WITH_METADATA else _CleanProResEncoder
                encoder = encoder_type(
                    ffmpeg_path=ffmpeg_path, profile=profile, framerate=fps,
                    output_path=encoded_path, input_transfer=transfer,
                )
                encoder.encode(
                    images=batch, mask=video_mask, audio=video_audio, metadata=metadata
                )
                if not cls.WITH_METADATA:
                    strip_mov_descriptive_metadata(encoded_path)
                output_path = publish_output(encoded_path, folder, name, counter)

            paths.append(str(output_path))
            previews.append({
                "filename": output_path.name, "subfolder": subfolder, "type": "output",
            })

        return IO.NodeOutput(
            images, paths,
            ui={"images": previews, "animated": (True,)},
        )


class SaveProReswithMetadata(_SaveProRes):
    WITH_METADATA = True

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_SaveProRes",
            display_name="Save ProRes with metadata",
            description="Exports one ProRes MOV per (B,H,W,C) list item, with optional metadata and audio.",
            category="KASKI/savers",
            search_aliases=["save prores", "save mov", "prores", "export video", "save video"],
            inputs=cls._inputs(),
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_input_list=True,
            is_output_node=True,
            outputs=[
                IO.Image.Output(display_name="images", is_output_list=True),
                IO.String.Output(display_name="filepath", is_output_list=True),
            ],
        )


class SaveProReswithoutMetadata(_SaveProRes):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_SaveProResWithoutMetadata",
            display_name="Save ProRes without metadata",
            description="Exports one MOV per (B,H,W,C) list item; strips descriptive container metadata and workflow.",
            category="KASKI/savers",
            search_aliases=["save prores", "save mov", "prores", "export video", "save video", "clean prores", "clean mov", "no workflow"],
            inputs=cls._inputs(),
            is_input_list=True,
            is_output_node=True,
            outputs=[
                IO.Image.Output(display_name="images", is_output_list=True),
                IO.String.Output(display_name="filepath", is_output_list=True),
            ],
        )


# Keep the original Python import name and serialized node ID usable.
SaveProRes = SaveProReswithMetadata
VIDEO_SAVER_NODES_LIST = [SaveProReswithMetadata, SaveProReswithoutMetadata]




