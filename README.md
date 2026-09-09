# KASKI Nodes

A compact collection of production-oriented custom nodes for [ComfyUI](https://github.com/Comfy-Org/ComfyUI).

KASKI Nodes is built around the less glamorous parts of real AI/VFX production: predictable naming, filename-aware loading, reusable API configuration, generation metadata, video conformation, temporal utilities, and small workflow primitives.

> Built for production pipelines rather than one-off demo workflows.

## Highlights

- Fully migrated to ComfyUI's current **V3 node API**
- Unified image API routing for **OpenAI GPT Image, Google Gemini / Nano Banana, ByteDance Seedream and Black Forest Labs FLUX.2**
- Reusable API settings that can feed multiple generators
- Production naming tools for shots and reusable reference assets
- Filename-aware image and video loading
- 8-bit / 16-bit PNG saving with ComfyUI workflow metadata, generation metadata and optional embedded reference images
- WAN/VACE resolution and frame-count helpers
- Video extension, shortening and temporal smoothing
- Autogrowing string assembly
- Small workflow utilities such as JSON fragments, number formatting and asynchronous delay
- ProRes MOV saving with audio retiming, optional alpha and production metadata
- Standalone browser-based PNG / MOV metadata inspector and documentation-PDF generator

All ComfyUI nodes are grouped below the `KASKI` category.

---

## Installation

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> KASKI-Nodes
```

Then restart ComfyUI.

You can also download the repository as a ZIP and extract it to:

```text
ComfyUI/custom_nodes/KASKI-Nodes
```

### Requirements

KASKI Nodes targets recent ComfyUI builds and uses current interfaces including:

- `comfy_api.latest`
- `comfy_api_nodes`
- V3 `IO.Schema`
- dynamic inputs
- autogrowing inputs
- ComfyUI's API-node authentication and proxy infrastructure

A current ComfyUI installation is strongly recommended.

Local utility nodes do not require external services. API generation requires the corresponding provider access to be configured through ComfyUI.

### ProRes / FFmpeg

The ProRes saver requires an FFmpeg build with ProRes support. KASKI resolves an available FFmpeg executable at runtime and can use `imageio-ffmpeg` as a convenient packaged source.

Recommended Python dependency:

```text
imageio-ffmpeg
```

A system FFmpeg installation may also be used, depending on the resolver configuration in `prores_saver.py`.

---

# Nodes

## Unified Image API

Category:

```text
KASKI/api-adaptions/image
```

The unified image API is intentionally a thin adapter over ComfyUI's built-in provider nodes.

KASKI owns the shared settings, routing and normalized outputs. Authentication, uploads, request execution, validation and response decoding remain inside `comfy_api_nodes`.

This keeps the KASKI layer small while still providing a common production interface across providers.

### KASKI Image API Settings

Creates a reusable settings object for the selected image backend.

Supported providers:

- OpenAI GPT Image
- Google Gemini / Nano Banana
- ByteDance Seedream
- Black Forest Labs FLUX.2

Available settings depend on the selected provider and model.

Current model families include:

**OpenAI**

- `gpt-image-2`
- `gpt-image-1.5`
- `gpt-image-1`

**Gemini**

- Gemini 3 Pro Image
- Nano Banana 2 / Gemini 3.1 Flash Image
- Nano Banana 2 Lite

**Seedream**

- Seedream 5.0 Pro
- Seedream 5.0 Lite

**FLUX.2**

- Flux.2 Pro
- Flux.2 Max

A single settings node can be connected to several generator nodes so model configuration stays synchronized across a workflow.

### KASKI Image API Generator

Routes a generation request to the backend selected by the connected settings object.

Inputs include:

- prompt
- settings
- seed
- optional IMAGE references
- optional mask
- optional compatible Gemini files

The exact provider behavior remains defined by ComfyUI's built-in API nodes.

Normalized outputs:

1. `image`
2. `thoughts`
3. `thought_image`
4. `prompt`
5. `modelName`
6. `seed`

The final three outputs are strings intended to connect directly to the KASKI PNG saver.

### Error behavior

The API adapter uses soft error handling.

If provider execution fails:

- the exception and traceback are printed to the ComfyUI console
- the generator returns a black placeholder image
- the error is surfaced through the `thoughts` output
- generation provenance fields are left empty

A black output should therefore be treated as an API failure and checked against the console log.

---

## PNG Saver

Category:

```text
KASKI/savers
```

### Save PNG with metadata

Saves ComfyUI IMAGE tensors as PNG without applying additional color transforms.

Features:

- 8-bit PNG
- 16-bit PNG
- batch saving
- standard ComfyUI workflow metadata
- standard ComfyUI execution-prompt metadata
- optional model name
- optional user prompt
- optional generation seed
- optional IMAGE batch containing the visual references used for generation
- each connected reference frame is embedded as its own 8-bit PNG inside the saved output PNG
- standard ComfyUI filename-prefix formatting tokens

Custom metadata keys:

```text
model_name
user_prompt
seed
```

Reference images are stored as private ancillary PNG chunks. This keeps the output image self-contained for documentation without changing the visible image data. If no reference IMAGE batch is connected, no reference-image chunks are written.

ComfyUI's own workflow metadata remains stored using its standard keys.

The saver intentionally performs quantization only. Input pixel values are assumed to already represent the desired display-ready image.

---

## PNG / MOV Metadata Toolkit

The repository also contains a standalone browser utility:

```text
KASKI_PNG_Metadata_Toolkit.html
```

It runs fully locally and supports both still-image and video documentation.

The single-file viewer automatically detects whether the dropped file is a PNG or MOV.

For PNG files it can display:

- saved image
- model name
- seed
- user prompt
- ComfyUI workflow
- ComfyUI execution prompt
- embedded reference images

For MOV files it can display:

- embedded video preview image
- model name
- seed
- user prompt
- ComfyUI workflow
- ComfyUI execution prompt
- reference filenames

The MOV reader avoids loading the complete video payload. It scans the top-level QuickTime box structure and reads only the `moov` box required for metadata and the embedded preview image. Large ProRes `mdat` payloads therefore do not need to pass through browser memory just to inspect documentation metadata.

The toolkit also provides separate folder-to-PDF modes for PNG and MOV files. Each source file produces a documentation page containing the relevant preview, metadata and references. Long prompts automatically continue onto additional pages.

No ComfyUI installation or server is required to use the HTML tool.

---

## ProRes Saver

Category:

```text
KASKI/savers
```

### Save ProRes

Saves a ComfyUI IMAGE batch as a QuickTime `.mov` using Apple ProRes via FFmpeg.

Inputs include:

- IMAGE batch interpreted as consecutive video frames
- optional AUDIO
- optional MASK
- ProRes profile
- output framerate
- filename prefix
- optional model name
- optional prompt
- integer seed
- autogrowing reference-filename STRING inputs

Available profiles:

```text
ProRes 422 Proxy
ProRes 422 LT
ProRes 422
ProRes 422 HQ
ProRes 4444
ProRes 4444 XQ
```

The 422 profiles use a 10-bit output path. ProRes 4444 supports optional alpha and uses a 10-bit 4:4:4 path. ProRes 4444 XQ is treated as a strict 12-bit target: the saver should error instead of silently pretending a 10-bit encode is 12-bit when the installed FFmpeg backend cannot provide the required path.

### Alpha

The optional MASK input follows normal Comfy semantics:

```text
0.0 / black  = transparent
1.0 / white  = visible
```

Alpha is only used by profiles that support it. If a MASK is connected while a 422 profile is selected, it is ignored with a warning. No automatic mask inversion is performed.

### Audio retiming

Audio is automatically conformed to the exact output video duration:

```text
video duration = frame_count / output_framerate
```

This is useful when an IMAGE sequence originated from a video at one framerate but is intentionally saved at another. The saver uses FFmpeg tempo processing to adjust audio duration while preserving pitch, then trims/pads to the exact video duration. Audio is written as uncompressed PCM in the MOV container.

### Video metadata and provenance

The saver stores generation information as QuickTime metadata, including:

```text
model_name
user_prompt
seed
reference_files
poster_frame_jpeg
```

`reference_files` is generated internally from the autogrowing STRING inputs and stored as a JSON array, preserving input order. This allows image, video and audio references to be documented without embedding potentially very large source media into the MOV.

Example:

```json
[
  "character_father_v3.png",
  "camera_reference.mov",
  "voice_reference.wav"
]
```

The saver also embeds a compact JPEG preview generated from video frame 0. This lets the standalone metadata toolkit display a representative video image even when the browser cannot decode ProRes itself. The preview is documentation-only and does not modify the encoded ProRes stream.

Standard ComfyUI workflow and execution-prompt metadata are retained as well unless metadata saving has been globally disabled.

---

## Filename-Aware Loaders

Category:

```text
KASKI/loaders
```

### Load Image with Filename

Loads an image through ComfyUI while preserving the original source filename as a separate STRING output.

Outputs:

1. `IMAGE`
2. `filename`
3. `MASK`

Image decoding and mask handling are delegated to ComfyUI's core image loader rather than reimplemented by KASKI.

Typical use:

```text
Load Image with Filename
    ├── IMAGE    → processing pipeline
    ├── filename → ID extraction / save-name construction
    └── MASK     → inpainting / mask workflow
```

### Load Video with Filename

Loads a ComfyUI VIDEO object and returns the source filename alongside it.

Outputs:

1. `VIDEO`
2. `filename`

The video object uses ComfyUI's current `VideoFromFile` implementation.

These loaders are useful whenever downstream naming, metadata, save paths or ID extraction must remain tied to the original source file.

---

## ID Tools

Category:

```text
KASKI/ID-Tools
```

The ID tools implement a small naming system for production shots and reusable reference assets.

### Reference IDs

Format:

```text
[project_]type_name_[artist_]version
```

Supported reference types:

```text
character
prop
location
material
```

Examples:

```text
character_father_v1
somat_character_father_v3
somat_character_father_KW_v3
project_location_kitchen_AB_vN
```

Available nodes:

| Node | Purpose |
|---|---|
| **Generate Reference ID** | Builds a reference ID from structured fields |
| **Extract Reference ID** | Extracts a reference ID from a larger string or filename |

Project names, reference names and artist codes must not contain spaces or underscores. Hyphens are supported.

`vN` represents an intentionally unresolved version.

### Shot IDs

Format:

```text
[project_]sh[number]_pipelineStep_[artist_]version
```

Examples:

```text
sh001_firstFrame_v1
somat_sh012_enhanced_v4
somat_sh012_firstFrame_KW_v5
project_sh120_cgi_AB_vN
```

Current pipeline-step vocabulary:

```text
firstFrame
lastFrame
ffToCleanup
lfToCleanup
Depth
Normal
cgi
Scribble
plate
plateToCleanup
notEnhanced
enhanced
```

Available nodes:

| Node | Purpose |
|---|---|
| **Generate Shot ID** | Builds a shot ID |
| **Modify Shot ID** | Replaces selected fields or increments an existing numeric version |
| **Extract Shot ID** | Extracts a valid shot ID from larger text or filenames |

Shot-number zero padding is configurable. The default is three digits.

### Naming quick guide

The repository includes a compact naming-convention reference:

```text
NamingConventions.pdf
```

The production convention follows:

```text
<project>_<type>_<name>_<artist>_<version>.<ext>
```

for reusable references and:

```text
<project>_sh###_<step>_<artist>_<version>.<ext>
```

for shot files.

---

## Input Conform

Category:

```text
KASKI/InputConform
```

### Min/Max Size

Calculates an output canvas size from minimum and maximum width/height constraints without modifying the input image.

The node attempts to preserve the original aspect ratio through proportional scaling.

If the active minimum and maximum constraints cannot all be satisfied through scaling alone, the result describes a canvas that expects the remaining space to be created through downstream padding.

A constraint value of `0` disables that constraint.

Outputs:

- `optimal_width`
- `optimal_height`

### Align Frames to Seconds

Calculates the smallest whole-second duration that can contain a frame sequence at the selected FPS, then returns the required frame count.

Outputs:

- `frames_to_lengthen_to`
- `length_in_seconds`
- `fps`

Useful for preparing clips for pipelines that expect whole-second durations.

### WAN Video Optimals

Calculates preferred WAN/VACE target parameters from an IMAGE sequence.

Outputs:

- `optimal_width`
- `optimal_height`
- `optimal_n_frames`

The node selects from predefined WAN/VACE resolution buckets and expands the temporal length to the next valid:

```text
4n + 1
```

frame count.

---

## Video Tools

Category:

```text
KASKI/videoTools
```

IMAGE batches are interpreted as temporal image sequences.

### Extend Video

Extends a sequence until it contains at least the requested number of frames.

Methods:

- `ping_pong`
- `repeat_last_frame`
- `repeat_from_start`

Sequences that are already long enough pass through unchanged.

### Shorten Video

Reduces a sequence to at most the requested number of frames.

Methods:

- `cut_end`
- `cut_beginning`
- `resample`

`resample` distributes the selected frames across the complete temporal span rather than simply trimming the sequence.

### Temporal Smoother

Analyzes local frame-to-frame motion and can rebuild the temporal sampling density around detected motion irregularities.

The current implementation uses RIFE optical flow for motion analysis.

Resampling methods:

- `rife`
- `blend`

The node produces:

1. resampled IMAGE sequence
2. human-readable motion table

The motion table exposes per-transition information including:

- measured motion
- local baseline
- motion ratio
- sensitivity-adjusted ratio
- target temporal intervals
- selected action

This node is experimental production tooling rather than a general-purpose frame-interpolation replacement.

---

## String Tools

Category:

```text
KASKI/stringtools
```

### JSON Key-Value String

Builds a JSON-style key/value fragment for assembling structured strings inside a workflow.

Supports optional nested wrapping.

### String Split at Symbol

Splits a string using a custom delimiter and returns the selected element by index.

### Join Strings

Joins multiple STRING inputs using a configurable delimiter.

The input list uses ComfyUI V3 autogrowing inputs and can expand dynamically.

### Number to String

Formats either an integer or float as a STRING.

Supports:

- integer / float mode
- zero padding
- configurable decimal places

---

## Async Utilities

Category:

```text
KASKI/async
```

### Async Delay

Passes an IMAGE through unchanged after a configurable asynchronous delay in milliseconds.

Useful for:

- staggering operations
- testing asynchronous workflows
- simple timing offsets

The delay uses `asyncio.sleep()` and does not intentionally block the entire event loop.

---

# Suggested Workflow Patterns

## Generation → Metadata → Save

```text
KASKI Image API Settings
          │
          ▼
KASKI Image API Generator
    ├── image ─────────────────────────────┐
    ├── prompt ───────────────┐            │
    ├── modelName ────────┐   │            │
    └── seed ──────────┐  │   │            │
                      ▼  ▼   ▼            ▼
                 Save PNG with metadata
```

Optional reference IMAGE batches can be connected to the saver so the exact visual references are carried inside the resulting PNG.

For video workflows, the same provenance idea is intentionally lighter-weight: the ProRes saver stores reference filenames as a JSON array instead of embedding potentially huge image/video/audio sources.

## Filename-Driven Shot Processing

```text
Load Image with Filename
    ├── IMAGE → processing pipeline
    └── filename
            ↓
      Extract Shot ID
            ↓
       Modify Shot ID
            ↓
     save-name construction
```

## Centralized API Configuration

```text
KASKI Image API Settings
    ├── KASKI Image API Generator
    ├── KASKI Image API Generator
    └── KASKI Image API Generator
```

One settings object can coordinate multiple generation branches.

## Production Reference Naming

```text
Generate Reference ID
        ↓
somat_character_father_KW_v3
        ↓
metadata / save path / review export
```

---

# Architecture

KASKI Nodes uses ComfyUI's V3 extension interface.

Each module exports a list of node classes, and the package-level extension collects them through a single `comfy_entrypoint()`.

Conceptually:

```text
__init__.py
    │
    ├── ASYNC tools
    ├── ID tools
    ├── image saver
    ├── ProRes saver
    ├── input conform
    ├── loaders
    ├── string tools
    ├── unified image API
    └── video tools
```

The package intentionally avoids legacy `NODE_CLASS_MAPPINGS` registration.

---

# Repository Structure

```text
KASKI-Nodes/
├── __init__.py
├── async_tools.py
├── id_tools.py
├── image_saver.py
├── prores_saver.py
├── input_conform.py
├── loaders.py
├── string_tools.py
├── unified_image_api.py
├── video_tools.py
│
├── external_libraries/
│   └── RIFE/
│
├── KASKI_PNG_Metadata_Toolkit.html
├── NamingConventions.pdf
├── README.md
└── .gitignore
```

---

# Design Philosophy

The package follows a few simple principles:

**Own workflow behavior, not upstream infrastructure.**  
Where ComfyUI already provides decoding, authentication, provider execution or transport logic, KASKI tries to reuse it instead of maintaining a parallel implementation.

**Keep production identity explicit.**  
Shot IDs, asset IDs, filenames and generation provenance should survive the workflow rather than becoming implicit knowledge. PNG outputs can carry their visual references directly; video outputs carry reference filenames and a compact preview image so documentation remains useful without turning master files into media archives.

**Small nodes should compose.**  
A filename loader, an ID parser, a string formatter and a metadata saver are individually simple. Their value comes from making larger workflows predictable.

**Prefer boring reliability over clever abstraction.**  
Most nodes exist because a production workflow needed a repeatable answer to a mundane problem.

---

# Development Status

KASKI Nodes is built around active production needs and recent ComfyUI APIs.

The unified API adapter deliberately reuses provider interfaces from `comfy_api_nodes`, including some upstream helper functions. These interfaces can change as ComfyUI evolves.

When updating ComfyUI, verify the API-adaption nodes before relying on them in unattended or production-critical workflows.

The local utility nodes are substantially less dependent on provider-specific upstream behavior.

Issues and focused pull requests are welcome.
