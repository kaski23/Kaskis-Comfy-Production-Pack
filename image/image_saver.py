import json
import os
import struct
import zlib

import numpy as np
import torch

import folder_paths
from comfy_api.latest import ComfyExtension, IO
from comfy.cli_args import args


# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REFERENCE_IMAGE_CHUNK = b"krFI"


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Build a PNG chunk: length | type | data | CRC."""
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF

    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", crc)
    )


def _png_text(keyword: str, value: str) -> bytes:
    """
    Create a PNG text chunk.

    Uses tEXt for Latin-1 text and iTXt for Unicode. This preserves
    ComfyUI's standard metadata keys while allowing Unicode prompts.
    """
    key = keyword.encode("latin-1")

    if not 1 <= len(key) <= 79 or b"\x00" in key:
        raise ValueError(f"Invalid PNG metadata keyword: {keyword!r}")

    try:
        text = value.encode("latin-1")
    except UnicodeEncodeError:
        # iTXt: keyword, compression flag, compression method,
        # language tag, translated keyword, UTF-8 text.
        payload = key + b"\x00\x00\x00\x00\x00" + value.encode("utf-8")
        return _png_chunk(b"iTXt", payload)

    return _png_chunk(b"tEXt", key + b"\x00" + text)


def inject_png_metadata(png_bytes: bytes, metadata: dict) -> bytes:
    """Insert metadata immediately after the PNG IHDR chunk."""
    if not metadata:
        return png_bytes

    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise ValueError("Invalid PNG data.")

    ihdr_length = struct.unpack(">I", png_bytes[8:12])[0]
    ihdr_end = 8 + 8 + ihdr_length + 4

    chunks = b"".join(
        _png_text(str(key), str(value))
        for key, value in metadata.items()
    )

    return png_bytes[:ihdr_end] + chunks + png_bytes[ihdr_end:]


def inject_reference_images(
    png_bytes: bytes,
    reference_pngs: list[bytes],
) -> bytes:
    """
    Embed reference images as private ancillary PNG chunks immediately
    before IEND.

    Each krFI chunk contains one complete 8-bit PNG file. Chunk order
    matches the frame order of the input_images tensors.
    """
    if not reference_pngs:
        return png_bytes

    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise ValueError("Invalid PNG data.")

    offset = 8

    while offset + 12 <= len(png_bytes):
        chunk_start = offset
        length = struct.unpack(">I", png_bytes[offset:offset + 4])[0]
        chunk_type = png_bytes[offset + 4:offset + 8]
        chunk_end = offset + 12 + length

        if chunk_end > len(png_bytes):
            raise ValueError("Invalid or truncated PNG chunk stream.")

        if chunk_type == b"IEND":
            embedded = b"".join(
                _png_chunk(_REFERENCE_IMAGE_CHUNK, ref_png)
                for ref_png in reference_pngs
            )
            return png_bytes[:chunk_start] + embedded + png_bytes[chunk_start:]

        offset = chunk_end

    raise ValueError("PNG contains no IEND chunk.")


# ---------------------------------------------------------------------------
# Tensor -> PNG
# ---------------------------------------------------------------------------

def _encode_png(image: np.ndarray, bit_depth: int) -> bytes:
    """
    Encode one HxWxC integer image into PNG bytes.

    No external image library is used. PNG scanline filter 0 is used,
    followed by zlib compression.
    """
    height, width, channels = image.shape

    color_types = {
        1: 0,  # Grayscale
        3: 2,  # RGB
        4: 6,  # RGBA
    }

    color_type = color_types[channels]

    # PNG stores 16-bit samples in big-endian byte order.
    dtype = np.dtype("u1") if bit_depth == 8 else np.dtype(">u2")
    image = np.ascontiguousarray(image.astype(dtype, copy=False))

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,  # Compression method
        0,  # Filter method
        0,  # Interlace method
    )

    compressor = zlib.compressobj(level=6)
    compressed = bytearray()

    for row in image:
        # Filter type 0: the scanline contains unmodified sample bytes.
        compressed.extend(compressor.compress(b"\x00" + row.tobytes()))

    compressed.extend(compressor.flush())

    # Split large compressed streams into valid-sized IDAT chunks.
    idat_chunks = b"".join(
        _png_chunk(b"IDAT", bytes(compressed[i:i + 1024 * 1024]))
        for i in range(0, len(compressed), 1024 * 1024)
    )

    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + idat_chunks
        + _png_chunk(b"IEND", b"")
    )


def tensor_to_png(
    image: torch.Tensor,
    bit_depth: int = 8,
) -> list[bytes]:
    """
    Convert a ComfyUI IMAGE tensor (B, H, W, C) into PNG byte strings.

    The input values are assumed to already be display-ready RGB values.
    No gamma, EOTF, color-space matrix or other color transform is applied.

    Supported channels: 1 (grayscale), 3 (RGB), 4 (RGBA).
    Supported bit depths: 8 and 16.

    Values are clamped to [0, 1] and quantized to the selected bit depth.
    NaN is mapped to 0, +inf to 1 and -inf to 0.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected IMAGE tensor (B, H, W, C), got {tuple(image.shape)}."
        )

    batch, height, width, channels = image.shape

    if batch < 1 or height < 1 or width < 1:
        raise ValueError("IMAGE tensor must not be empty.")

    if channels not in (1, 3, 4):
        raise ValueError(
            f"Unsupported channel count: {channels}. Expected 1, 3 or 4."
        )

    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16.")

    max_value = (1 << bit_depth) - 1

    # Quantization only. No color management.
    quantized = (
        image.detach()
        .to(dtype=torch.float32)
        .nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)
        .clamp(0.0, 1.0)
        .mul(max_value)
        .round()
        .to(torch.int32)
        .cpu()
        .numpy()
    )

    return [
        _encode_png(frame, bit_depth)
        for frame in quantized
    ]


def _list_value(values, index, default=None):
    """Broadcast a single widget value across a ComfyUI images list."""
    if values is None:
        return default
    if not isinstance(values, list):
        return values
    return values[min(index, len(values) - 1)] if values else default


class _SavePNG(IO.ComfyNode):
    """Common list handling, encoding, path allocation and UI results."""

    WITH_METADATA = False

    @classmethod
    def _inputs(cls):
        inputs = [
            IO.Image.Input("images"),
            IO.String.Input(
                "filename_prefix", default="ComfyUI",
                tooltip="Filename prefix. Supports ComfyUI formatting tokens.",
            ),
            IO.Combo.Input("bit_depth", options=["8-bit", "16-bit"], default="16-bit"),
        ]
        if cls.WITH_METADATA:
            inputs += [
                IO.Image.Input(
                    "input_images", optional=True,
                    tooltip="Reference images or batches embedded in each exported PNG; a reference list does not create extra files.",
                ),
                IO.String.Input("model_name", default="", optional=True),
                IO.String.Input("user_prompt", default="", multiline=True, optional=True),
                IO.Int.Input("seed", default=-1, optional=True),
                IO.Boolean.Input(
                    "save_workflow", default=True, display_name="Save Workflow",
                    tooltip="Embed ComfyUI execution prompt and workflow. Other custom metadata remains enabled.",
                ),
            ]
        return inputs

    @classmethod
    def execute(
        cls, images, input_images=None, filename_prefix=None, bit_depth=None,
        model_name=None, user_prompt=None, seed=None, save_workflow=None,
    ) -> IO.NodeOutput:
        # INPUT_IS_LIST: each item in images is one tensor/batch to save.
        # input_images is a collection of references, never a second output axis.
        if not isinstance(images, list):
            images = [images]
        if not images:
            raise ValueError("No IMAGE inputs were provided.")

        reference_pngs = []
        if cls.WITH_METADATA and not args.disable_metadata:
            for batch in input_images or []:
                if batch is not None:
                    reference_pngs.extend(tensor_to_png(batch, bit_depth=8))

        results = []
        for index, batch in enumerate(images):
            prefix = _list_value(filename_prefix, index, "ComfyUI")
            depth_name = _list_value(bit_depth, index, "16-bit")
            if depth_name not in ("8-bit", "16-bit"):
                raise ValueError(f"Unsupported PNG bit depth: {depth_name}")
            pngs = tensor_to_png(batch, bit_depth=8 if depth_name == "8-bit" else 16)
            metadata = cls._metadata(
                _list_value(model_name, index, ""),
                _list_value(user_prompt, index, ""),
                _list_value(seed, index, -1),
                _list_value(save_workflow, index, True),
            )
            output_dir = folder_paths.get_output_directory()
            folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(
                prefix, output_dir, batch.shape[2], batch.shape[1]
            )
            for batch_number, png_bytes in enumerate(pngs):
                if metadata:
                    png_bytes = inject_png_metadata(png_bytes, metadata)
                if reference_pngs:
                    png_bytes = inject_reference_images(png_bytes, reference_pngs)
                base = name.replace("%batch_num%", str(batch_number))
                while True:
                    filename = f"{base}_{counter:05}_.png"
                    try:
                        with open(os.path.join(folder, filename), "xb") as output:
                            output.write(png_bytes)
                        break
                    except FileExistsError:
                        counter += 1
                results.append({"filename": filename, "subfolder": subfolder, "type": "output"})
                counter += 1

        return IO.NodeOutput(images, ui={"images": results})

    @classmethod
    def _metadata(cls, model_name, user_prompt, seed, save_workflow):
        if not cls.WITH_METADATA or args.disable_metadata:
            return {}
        metadata = {}
        if save_workflow:
            if cls.hidden.prompt is not None:
                metadata["prompt"] = json.dumps(cls.hidden.prompt, ensure_ascii=False)
        extra_pnginfo = cls.hidden.extra_pnginfo
        if extra_pnginfo:
            for key, value in extra_pnginfo.items():
                if not save_workflow and key in ("workflow", "prompt"):
                    continue
                metadata.setdefault(key, json.dumps(value, ensure_ascii=False))
        if model_name:
            metadata["model_name"] = str(model_name)
        if user_prompt:
            metadata["user_prompt"] = str(user_prompt)
        metadata["seed"] = str(seed).strip()
        return metadata


class SavePNGwithMetadata(_SavePNG):
    WITH_METADATA = True

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_SaveImage",
            display_name="Save PNG with metadata",
            description="Saves 8-bit or 16-bit PNGs without color transforms.",
            category="KASKI/savers",
            search_aliases=["save png", "save image", "export png"],
            inputs=cls._inputs(),
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_input_list=True,
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images", is_output_list=True)],
        )


class SavePNGwithoutMetadata(_SavePNG):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="KASKI_SaveImageWithoutMetadata",
            display_name="Save PNG without metadata",
            description="Saves PNGs with only IHDR, IDAT and IEND chunks; never embeds workflow or references.",
            category="KASKI/savers",
            search_aliases=["save png", "save image", "export png", "clean png", "without metadata", "no metadata", "no workflow"],
            inputs=cls._inputs(),
            is_input_list=True,
            is_output_node=True,
            outputs=[IO.Image.Output(display_name="images", is_output_list=True)],
        )


IMAGE_SAVER_NODES_LIST = [
    SavePNGwithMetadata, 
    SavePNGwithoutMetadata
]


