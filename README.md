# KASKI Nodes

Production-oriented custom nodes for [ComfyUI](https://github.com/Comfy-Org/ComfyUI), focused on practical AI/VFX workflows: API image generation, production naming, filename-aware loading, metadata-safe saving, video conformation, JSON assembly and small workflow utilities.

All nodes are available below the `KASKI` category.

## Installation

Clone the repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/kaski23/Kaskis_Comfy_Nodes_v2.git
```

Then restart ComfyUI.

A recent ComfyUI build is recommended. API nodes use ComfyUI's current V3 API and built-in provider infrastructure.

### FFmpeg

`Save ProRes` requires FFmpeg with ProRes support. `imageio-ffmpeg` can be used as a packaged FFmpeg source:

```bash
pip install imageio-ffmpeg
```

A compatible system FFmpeg installation can also be used.

---

# Nodes

## Image API

Category: `KASKI/api-adaptions/image`

### KASKI Image API Settings

**Problem:** Provider-specific image nodes expose different settings, making it cumbersome to switch models or keep several generator branches configured consistently.

Creates one reusable settings object for the KASKI image generator. The visible controls change automatically with the selected backend and model.

Supported backends currently include:

- OpenAI GPT Image
  - `gpt-image-2.5-flare`
  - `gpt-image-2.5-sunburst`
  - `gpt-image-2`
- Google Gemini / Nano Banana
  - Gemini 3 Pro Image
  - Nano Banana 2
  - Nano Banana 2 Lite
- ByteDance Seedream
  - Seedream 5.0 Pro
  - Seedream 5.0 Lite
- Black Forest Labs FLUX.2
  - Flux.2 Pro
  - Flux.2 Max

Features:

- Dynamic model-specific controls
- One settings output can feed multiple generator nodes
- Shared Gemini system prompt
- OpenAI multi-image output count
- Gemini response modality controls
- Seedream watermark and thinking controls
- Provider execution remains inside ComfyUI's built-in API nodes

### KASKI Image API Generator

**Problem:** Different image providers return different outputs and require different execution paths, which makes provider-independent workflows difficult to build.

Routes a request through the backend selected in `KASKI Image API Settings` and normalizes the result.

Inputs:

- `prompt`
- `settings`
- `seed`
- optional `images`
- optional `mask`
- optional Gemini-compatible `files`

Outputs:

1. `image`
2. `thoughts`
3. `thought_image`
4. `prompt`
5. `modelName`
6. `seed`

Features:

- Text-to-image and reference-image generation
- IMAGE batches can be used as reference collections where supported
- Optional masking where supported by the selected provider
- Gemini file input support
- Normalized outputs across all supported providers
- Prompt, model and seed outputs can be connected directly to KASKI saver nodes

The `seed` primarily participates in ComfyUI execution/cache behavior. Whether a provider receives or uses a seed depends on its underlying ComfyUI implementation.

If an API call fails, the node prints the traceback to the ComfyUI console, returns a black placeholder image and exposes the error through `thoughts`.

---

## JSON Tools

Category: `KASKI/jsontools`

### Generate DICT from Key-Value-Pair

**Problem:** Building structured prompts as JSON strings becomes fragile once values themselves contain nested objects, quotes or multiple sub-sections.

Builds a real Python/ComfyUI `DICT` from one key and either a string value or one or more nested DICTs.

Modes:

#### `STRING`

Inputs:

- `key`
- multiline `value`

Example:

```text
key   = prompt
value = A woman walking through a forest
```

Output:

```python
{
    "prompt": "A woman walking through a forest"
}
```

#### `DICT`

The string field is replaced by an autogrowing list of DICT inputs.

Connected DICTs are merged in input order and placed below `key`.

Example:

```python
dict1 = {"character": "Anna"}
dict2 = {"camera": "35mm"}
dict3 = {"lighting": "soft"}
```

with:

```text
key = shot
```

produces:

```python
{
    "shot": {
        "character": "Anna",
        "camera": "35mm",
        "lighting": "soft"
    }
}
```

Features:

- Dynamic `STRING` / `DICT` interface
- Up to 64 nested DICT inputs
- Automatic merging of connected DICTs
- Later DICTs overwrite earlier values when the same key occurs
- Empty key returns an empty DICT

Use this node recursively to build larger structured dictionaries before serializing them.

### Generate JSON from DICT

**Problem:** Structured ComfyUI dictionaries eventually need to become valid JSON text for APIs, prompts or other string-based nodes.

Serializes a DICT into a valid JSON string.

Inputs:

- `data` — DICT to serialize
- `pretty` — formatted multiline JSON when enabled; compact JSON when disabled

Features:

- Correct JSON quoting and escaping
- Supports nested dictionaries automatically
- Preserves Unicode characters directly
- Compact mode removes unnecessary whitespace

Typical workflow:

```text
Generate DICT from Key-Value-Pair
            │
            ├── nested DICT builders
            │
            ▼
Generate JSON from DICT
            │
            ▼
         JSON string
```

---

## String Tools

Category: `KASKI/stringtools`

### JSON Key-Value String — Legacy

**Problem:** Older workflows may already assemble JSON manually from string fragments and need to remain loadable.

Creates a JSON-style key/value string fragment and optionally wraps the value as a nested object.

This node is retained for compatibility. New workflows should use `Generate DICT from Key-Value-Pair` together with `Generate JSON from DICT`, which avoids manual JSON-string assembly.

### String Split at Symbol

**Problem:** Production filenames and IDs often contain several fields separated by a known delimiter, but only one field is needed downstream.

Splits `text` by `delimiter` and returns the element at `index`.

Features:

- Custom delimiter
- Zero-based index
- Returns an empty string when the requested index does not exist
- Raises an error when the delimiter is empty

Example:

```text
text      = somat_sh012_firstFrame_v3
delimiter = _
index     = 1
```

Output:

```text
sh012
```

### Join Strings

**Problem:** Building filenames, IDs, paths or prompt fragments from a variable number of strings otherwise requires chains of concatenation nodes.

Joins an autogrowing list of STRING inputs using one delimiter.

Features:

- 2–50 STRING inputs
- Inputs grow automatically as connections are added
- Configurable delimiter
- Preserves input order

### Number to String

**Problem:** Numeric values often need deterministic text formatting before they can be used inside filenames, IDs or prompts.

Converts either an integer or float to a STRING.

Features:

- `INT` and `FLOAT` modes
- Zero padding
- Configurable decimal places for floats

Example:

```text
42 + zero_padding 4
```

becomes:

```text
0042
```

---

## ID Tools

Category: `KASKI/ID-Tools`

KASKI IDs are intended for predictable production naming.

### Generate Reference ID

**Problem:** Reusable assets such as characters, props and locations need consistent identifiers that can safely travel through filenames and workflows.

Builds a reference ID from structured fields.

Format:

```text
[project_]type_name_[artist_]version
```

Reference types:

- `character`
- `prop`
- `location`
- `material`

Examples:

```text
character_father_v1
somat_character_father_v3
somat_character_father_KW_v3
```

Features:

- Optional project name
- Optional artist code
- Numeric version or unresolved `vN`
- Validates components against the naming convention
- Letters, numbers and hyphens are allowed inside components; spaces and underscores are not

### Extract Reference ID

**Problem:** A valid reference ID is often embedded inside a longer filename or path and needs to be recovered without manually parsing the string.

Searches `text` for a valid KASKI reference ID.

`fail_if_not_found`:

- `True` — raises an error when no ID is found
- `False` — passes the original input string through

### Generate Shot ID

**Problem:** Shot files need predictable names that encode shot number, pipeline stage, artist and version consistently.

Builds a shot ID.

Format:

```text
[project_]sh[number]_pipelineStep_[artist_]version
```

Examples:

```text
sh001_firstFrame_v1
somat_sh012_enhanced_v4
somat_sh012_firstFrame_KW_v5
```

Pipeline stages:

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

Features:

- Optional project name
- Configurable shot-number zero padding
- Optional artist code
- Numeric version or unresolved `vN`

### Modify Shot ID

**Problem:** Existing shot IDs often need one field changed without rebuilding the entire identifier manually.

Accepts an existing valid Shot ID and selectively modifies it.

Features:

- Keep or replace project name
- Keep or replace shot number
- Keep or replace pipeline stage
- Keep or replace artist code
- Keep or increment the numeric version
- Configurable shot-number padding
- Invalid incoming IDs pass through unchanged

### Extract Shot ID

**Problem:** Shot IDs are commonly embedded inside filenames, paths or longer strings and need to be isolated reliably.

Searches `text` for a valid KASKI Shot ID.

`fail_if_not_found`:

- `True` — raises an error when no ID is found
- `False` — passes the original string through

A compact naming reference is also included in `NamingConventions.pdf`.

---

## Filename-Aware Loaders

Category: `KASKI/loaders`

### Load Image with Filename

**Problem:** ComfyUI image workflows normally work with the decoded image, while the original source filename is often needed separately for naming, metadata or ID extraction.

Loads an image and returns:

1. `image`
2. original `filename`
3. `mask`

Features:

- Uses ComfyUI's standard image decoding
- Preserves the basename as STRING
- Supports normal ComfyUI image upload/selection
- Preserves standard mask behavior

### Load Video with Filename

**Problem:** Video workflows frequently need the source filename alongside the VIDEO object for metadata, naming or downstream bookkeeping.

Loads a video and returns:

1. `video`
2. original `filename`

The VIDEO object uses ComfyUI's current file-backed video implementation.

---

## Input Conform

Category: `KASKI/InputConform`

### Min/Max Size

**Problem:** An image often needs a target canvas that respects minimum and maximum resolution limits without manually calculating aspect-ratio-safe dimensions.

Calculates recommended output width and height from the input image dimensions and four optional constraints.

Inputs:

- `min_width`
- `min_height`
- `max_width`
- `max_height`

Set any constraint to `0` to disable it.

Features:

- Preserves source aspect ratio whenever the constraints allow it
- Does not resize the image itself
- When minimum and maximum limits cannot all be satisfied through scaling, the result assumes remaining minimum size will be created through downstream padding

Outputs:

- `optimal_width`
- `optimal_height`

### Align Frames to Seconds

**Problem:** Some video pipelines require clips whose duration lands on whole seconds, while a source frame count may not.

Calculates the smallest whole-second duration that contains the source sequence at the selected FPS.

Outputs:

- `frames_to_lengthen_to`
- `length_in_seconds`
- `fps`

Example at 24 fps:

```text
24 frames → 24 frames / 1 second
25 frames → 48 frames / 2 seconds
```

### WAN Video Optimals

**Problem:** WAN/VACE workflows require suitable resolution buckets and temporal lengths following the `4n + 1` frame rule.

Inspects an IMAGE sequence and calculates recommended WAN/VACE parameters without modifying the sequence.

Resolution buckets:

```text
480 × 832
832 × 480
512 × 512
768 × 768
1024 × 1024
1280 × 720
720 × 1280
```

Outputs:

- `optimal_width`
- `optimal_height`
- `optimal_n_frames`

The resolution selection considers aspect ratio, dimension difference and scaling cost. Frame count is expanded to the nearest valid `4n + 1` value at or above the source length.

---

## Video Tools

Category: `KASKI/videoTools`

IMAGE batches are interpreted as ordered video frames.

### Extend Video

**Problem:** A generated or processed sequence may contain fewer frames than a downstream video model or conform step requires.

Extends the sequence until it contains at least `n_frames`.

Methods:

- `ping_pong` — continues by reversing and replaying the sequence
- `repeat_last_frame` — holds the final frame
- `repeat_from_start` — loops frames from the beginning

Sequences already long enough pass through unchanged.

### Shorten Video

**Problem:** A sequence may contain more frames than a downstream model, duration or delivery format allows.

Reduces the sequence to at most `n_frames`.

Methods:

- `cut_end` — keeps the first frames
- `cut_beginning` — keeps the last frames
- `resample` — distributes retained frames across the complete temporal span

Sequences already short enough pass through unchanged.

### Temporal Smoother

**Problem:** AI-generated video can contain irregular apparent motion where individual frame transitions move much more or less than their local temporal neighborhood.

Analyzes frame-to-frame motion with RIFE optical flow and can rebuild the temporal sampling density around motion irregularities.

Inputs:

- `images`
- `resample`
- `resample_method`
  - `rife`
  - `blend`
- `sensitivity`

Sensitivity:

- `0` effectively disables correction strength
- `1` follows the measured motion ratio directly
- values above `1` increase correction strength

Outputs:

1. processed IMAGE sequence
2. `motion_table`

The motion table reports frame-level analysis including measured motion, local baseline, motion ratio, sensitivity-adjusted ratio, target temporal intervals and the selected action (`COLLAPSE`, `KEEP` or `INSERT`).

Disable `resample` to run analysis without modifying the sequence.

This is specialized temporal repair tooling, not a general frame-interpolation replacement.

---

## Savers

Category: `KASKI/savers`

### Save PNG with metadata

**Problem:** Production images need predictable high-bit-depth output while preserving enough generation metadata to reconstruct or document how they were created.

Saves IMAGE tensors directly as PNG.

Features:

- 8-bit or 16-bit PNG
- Batch saving
- No additional gamma or color-space transform
- Standard ComfyUI workflow metadata
- Standard ComfyUI execution-prompt metadata
- Optional `model_name`
- Optional `user_prompt`
- Optional `seed`
- Optional reference IMAGE batch
- Each reference frame is embedded into the PNG as its own 8-bit PNG payload
- Standard ComfyUI filename-prefix formatting tokens
- Returns the input IMAGE batch for downstream use

The node assumes input pixel values already represent the desired display-ready image.

### Save ProRes

**Problem:** IMAGE sequences need a production-friendly MOV export with controlled ProRes profile, correct timing, optional alpha/audio and retained generation provenance.

Encodes an IMAGE batch as Apple ProRes through FFmpeg.

Profiles:

```text
ProRes 422 Proxy
ProRes 422 LT
ProRes 422
ProRes 422 HQ
ProRes 4444
ProRes 4444 XQ
```

Features:

- Configurable output framerate
- Optional AUDIO
- Optional MASK as alpha
- Autogrowing reference-filename inputs
- Optional model, prompt and seed metadata
- Standard ComfyUI workflow/execution metadata
- Embedded JPEG poster frame for documentation tools
- Returns both the IMAGE sequence and saved filepath

Alpha:

```text
black / 0.0 = transparent
white / 1.0 = visible
```

Alpha is only used by profiles that support it. A mask connected to a 422 profile is ignored with a warning.

Audio is automatically retimed to:

```text
duration = frame_count / framerate
```

Pitch is preserved, then the result is trimmed or padded to the exact video duration. Audio is stored as uncompressed PCM.

The 422 profiles use a 10-bit path. ProRes 4444 supports alpha. ProRes 4444 XQ is treated as a strict 12-bit target and errors when the installed FFmpeg path cannot provide the required encode instead of silently falling back.

---

## Async Utilities

Category: `KASKI/async`

### Async Delay

**Problem:** Some workflows need a simple non-blocking timing offset between branches or operations.

Passes an IMAGE through unchanged after `delay` milliseconds.

Features:

- Configurable delay
- Uses `asyncio.sleep()`
- Does not intentionally block the complete event loop
- IMAGE data itself is unchanged

---

# Metadata Toolkit

The repository includes:

```text
KASKI_PNG_MOV_Metadata_Toolkit.html
```

**Problem:** Generation metadata embedded in production outputs is useful only if it can be inspected and documented without reopening ComfyUI.

The standalone HTML tool runs locally in a browser and supports PNG and MOV files.

For PNG it can inspect:

- output image
- model
- seed
- user prompt
- ComfyUI workflow
- ComfyUI execution prompt
- embedded reference images

For MOV it can inspect:

- embedded poster frame
- model
- seed
- user prompt
- ComfyUI workflow
- ComfyUI execution prompt
- reference filenames

It can also generate documentation PDFs from folders of PNG or MOV outputs.

No ComfyUI server is required to use the metadata toolkit.
