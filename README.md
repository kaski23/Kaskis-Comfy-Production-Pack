# Kaski’s Comfy Production Pack

## What the pack is for

With 40 shots and five takes each, I want to switch models, run variations in parallel and still find the right prompt three weeks later. It puts ComfyUI’s APIs in one workflow, keeps references and take names together, and gets shots to the frame count the edit needs. `final_final_for_real.mov` can retire.

Drop the pack into `ComfyUI/custom_nodes/`, restart ComfyUI and look under `KASKI`. The pack adds HTTP 402 to ComfyUI’s retryable API status codes. To inspect PNGs and MOVs locally or export a folder as a PDF, open the [metadata toolkit](tools_docs/KASKI_PNG_MOV_Metadata_Toolkit.html). You'll find the button for that in the top bar.

Also, try shaking a connected node at some point... ;)

## The packages

### API Adaptions

Compare providers in the same graph and send generation settings straight to your saver. Controls and seed repeatability depend on the model. The video adapters’ `error` output holds a failure message or `NO ERRORS TODAY - SUCCEEDED`.

#### KASKI Unified Image API Settings

Configure OpenAI GPT Image, Gemini, Seedream or FLUX.2 once and share `settings` across generators. On Gemini, `system_prompt` sets common instructions and `response_modalities` selects the output types.

#### KASKI Unified Image API

Connect prompt, seed and `settings`. All frames on `images` become references for one request. Prompt, seed and settings lists each use their first entry. `mask` goes to OpenAI, `files` to Gemini. Prompt, model and seed outputs feed the saver. A black placeholder carries the error in `thoughts`.

#### KASKI Text to Video API

For a motion test, enter a prompt and seed. Switch between Seedance, Kling and Gemini Omni in the model dropdown to try the same shot.

#### KASKI First / Last Frame to Video API

Prompt the motion and connect a start frame. Add `last_frame` for a target pose. Kling 4.0 accepts either endpoint or both. Kling 3.0 Omni takes an end frame or up to six additional reference images.

#### KASKI Reference to Video API

Guide the look or motion with one still per image socket. In Kling Edit, the first video is the clip to edit. Pick the model to match your references.

| Model or mode | Max. images | Videos | Max. audio clips |
| --- | ---: | --- | ---: |
| Seedance 2.5 | 30 | Up to 10 | 10 |
| Seedance 2.0 | 9 | Up to 3 | 3 |
| Gemini Omni | 14 | Up to 3 | 0 |
| Kling 3.0 Omni · Image References | 7 | 0 | 0 |
| Kling 3.0 Omni · Video Reference / Edit Video | 4 | Exactly 1 | 0 |
| Kling 4.0 | 10 | Up to 5, including the base clip in Edit mode | 0 |

#### KASKI Video Enhance API

Connect a finished `VIDEO` and pick HitPaw or Topaz for restoration or upscaling. Choose `Topaz · Interpolation Only` for extra frames. The `prompt` input applies to Topaz Astra 2.

#### KASKI Seedance 2.5 Draft

Audition up to 32 parallel takes at 480p. With `n_runs = 3`, seed `40` and `run_delay_ms = 100`, seeds 40, 41 and 42 are scheduled 100 ms apart. A delay of `0` launches them together. Video, task ID and error lists line up by index.

#### KASKI Seedance 2.5 Draft to Final Video

Pick the take you want, connect its `draft_task_id` and choose the final resolution (default `1080p`). The ID tells Seedance which draft to finish.

#### ElevenLabs Voice Selector (KASKI)

Reuse a character’s voice across takes. Select from the local catalogue or override the dropdown with a non-empty `alternative_voice_id`.

#### ElevenLabs JSON to Dialogue (KASKI)

Set `inputs` to the number of voice sockets you need. Connect a selector or clone to each. In `dialogue_json`, name the socket in `voice_id` and list turns in playback order. Reuse a socket whenever its speaker returns.

```json
[
  {"text": "Lights up.", "voice_id": "voice1"},
  {"text": "Rolling.", "voice_id": "voice2"}
]
```

### ID Tools

Keep names consistent from ingest to delivery. Blank or `NONE` omits project or artist. Version `-1` generates `vN`. Extract finds the first ID, strips suffixes and fills missing versions with `vN`. Disable `fail_if_not_found` to pass unmatched text through. Modify takes the bare ID. [Naming conventions](tools_docs/NamingConventions.pdf)

#### Generate Reference ID

Pick a type (`location`, `prop`, `character` or `material`), name and version. Add project and artist for `spot_prop_Suitcase_KK_v1`. Use hyphens within names. The four types are reserved as project prefixes.

#### Modify Reference ID

For a handoff, enter the new artist code, `KEEP` to retain it or `NONE` to remove it. `Increment` advances numeric versions. `vN` stays open.

#### Extract Reference ID

Recover the asset ID from an incoming file. `spot_prop_Suitcase_KK_v2_ref_00001_.png` yields `spot_prop_Suitcase_KK_v2`.

#### Generate Shot ID

Set shot number, stage and version for the saver. With `shot_no_zero_padding = 3`, shot 7 becomes `sh007_firstFrame_v1`. Padding is a minimum width.

#### Modify Shot ID

Move a shot from `plate` to `enhanced`. `KEEP` retains text fields and `shot_no = -1` preserves the original number and padding. Choose `Increment` for a version bump.

#### Extract Shot ID

Recover the shot name on load. `sh007_plate_v2_00001_.png` gives `sh007_plate_v2` for the next pass.

#### Generate Sequence ID

Enter `100` for the whole sequence or `100-01` for its first shot. With three-digit padding those become `seq100_rendered_v1` and `seq100-001_rendered_v1`. The shot uses the width of the formatted sequence number.

#### Modify Sequence ID

Change the stage or version while keeping the sequence and shot. With `sequence_no` set to `KEEP`, changing the stage takes `seq100-01_plate_v1` to `seq100-001_rendered_v1`. Enter a plain sequence number to drop the shot part.

#### Extract Sequence ID

Read the sequence ID from a delivery. `seq100-01_plate_v3_review.mov` gives `seq100-01_plate_v3`. Modify Sequence ID can then normalize the padding.

### Loaders

Load the source filename alongside the media so downstream nodes can name outputs and record their references.

#### Load Image with Filename

Select or upload a still. Send image and mask into processing and `filename` to a saver or ID node. It includes the extension.

### Input Conform

Calculate the dimensions and frame counts a model or delivery spec requires. Downstream resize, padding and video nodes apply those targets.

#### Min/Max Size

Enter size limits for each axis. `0` disables a limit. The node keeps the aspect ratio where possible. A 2000 × 500 plate capped at 1920 wide scales to 1920 × 480. With a minimum height of 720, the recommended canvas is **1920 × 720**. Add the missing 240 pixels as vertical padding downstream.

#### Align Frames to Seconds

Use this when the edit gives you frames and the API asks for whole seconds. Take the larger of `n_frames` and `minimum_length_frames`, divide by `fps` and round up. Multiply those whole seconds by `fps` and round up again for the target frame count. Send `length_in_seconds` to the API and `frames_to_lengthen_to` to Extend Video.

| Source frames | Minimum frames | FPS | Whole seconds | Target frames |
| ---: | ---: | ---: | ---: | ---: |
| 60 | 72 | 24 | 3 | 72 |
| 73 | 72 | 24 | 4 | 96 |
| 96 | 72 | 23.976 | 5 | 120 |

At 23.976 fps, 96 frames run just past four seconds. That’s why the last row jumps to five.

#### WAN Video Optimals

Connect a frame batch. WAN/VACE preset selection weighs dimensions and aspect ratio and penalizes downscaling. Frame count rounds up to the next valid `4n + 1`. A 1280 × 720 clip with 96 frames returns **1280 × 720, 97 frames**.

### Image

Prepare references and masks for generation, fix pixel values and save stills with their source material.

#### Create Image List

Wire references in order. A two-frame batch plus a portrait still yields three separate items, each at its original resolution.

#### Image Channel Math

Each RGB field is a per-pixel expression over equal-sized images. `r1` is red from image 1. `c1` follows the current output channel. Put `mix(c1, c2, 0.25)` in all three fields for a 75/25 mix, or `clamp(c1, 0, 1)` to clip overshoot. A still can feed every frame of a batch. Output values are unclipped by default.

#### Inpaint/Outpaint Images with Color

Each `extend_*` input is a percentage of the original dimension. For more headroom, `extend_top = 20` adds 216 pixels above a 1080-pixel-high plate. The color fills the border and masked area. The output mask marks both.

#### Save PNG with metadata

Save display-ready frames at 8 or 16 bits with workflow, prompt, model and seed. Pixel values are clipped to 0–1 without a color transform. All `input_images` are embedded as 8-bit references in every PNG. `filename_prefix` accepts subfolders and ComfyUI formatting tokens.

### Video

Load footage, fix its timing and export a ProRes MOV for editorial. Frame tools take `IMAGE` batches. RIFE is bundled for motion analysis and interpolation.

#### Load Video with Filename

Load a clip. Use `video` for an API or `images` for frame processing. Filename and alpha are separate outputs. Footage without alpha gets an opaque mask.

#### Load Video with KASKI Metadata

Reconstruct a take from its MOV tags. Read prompt, model, seed and reference filenames. `workflow_json` holds the editor layout and graph. `workflow_prompt_json` holds the API execution graph. A missing seed reads as `-1`.

#### Extend Video

Set `n_frames` to the minimum length. Here’s what happens when you extend `A B C` to five frames. With `interpolated`, `scale` sets RIFE’s internal processing resolution.

| Method | Five output frames |
| --- | --- |
| `repeat_last_frame` | A, B, C, C, C |
| `repeat_from_start` | A, B, C, A, B |
| `ping_pong` | A, B, C, B, A |
| `interpolated` | A, halfway from A to B, B, halfway from B to C, C |

#### Shorten Video

Cap the frame count with `n_frames`. For `A B C D E` shortened to three, `cut_end` keeps `A B C`, `cut_beginning` keeps `C D E` and `resample` picks `A C E`.

#### Temporal Smoother

For a stuttering pan, disable `resample` and inspect `motion_table`. Each transition’s motion is compared with the median of nearby transitions. At `sensitivity = 1`, a duplicate in a moving shot reads `COLLAPSE`, a typical step reads `KEEP`, and twice the local motion reads `INSERT 1`. `0` disables correction.

Enable `resample` to apply it. `rife` interpolates motion and `blend` crossfades. `scale_analysis` sets RIFE’s internal scale for measuring motion, `scale_fill` for generating frames. The output frame count can change.

#### Save Video as ProRes

Connect frames, set FPS and pick a ProRes profile. Match `input_transfer` to the source (`sRGB` or `Rec.709`). Audio is retimed at constant pitch. At 25 fps, 100 frames make a four-second MOV, so a five-second voice-over plays at 1.25× speed.

For alpha, choose 4444 or 4444 XQ and connect `mask` (white is opaque). `reference_files` stores filenames in order alongside prompt, model and seed. Requires FFmpeg with `prores_ks` and 12-bit support for XQ.

### Audio

Use a recorded line’s duration to work out how much picture the edit needs.

#### Get Audio Length

Connect the voice-over as `AUDIO`. `length_seconds` divides samples by sample rate, so 120,000 samples at 48 kHz give 2.5 seconds.

### Flow Control

Switch optional branches and manage lists of takes or references. Each gate’s `enabled` switch controls all its channels. When off, `return None` supplies empty values and `return executionBlocker` stops downstream execution. A closed gate doesn’t request its channel inputs.

#### Bundler

Tidy up a cable run. Connect image, mask and ID on the left, then use the matching outputs on the right. Sockets grow as you wire them.

#### First Valid Input

Wire sources in priority order, fallback last. Returns the first value other than `None` or an empty string. `0` and `False` count. All empty gives `None`.

#### 1-Channel Gate

Before an optional upscale, select `return executionBlocker`. Enable the gate when that branch should run.

#### 2-Channel Gate

Wire image and mask separately. `enabled` sends both to the inpainting branch together.

#### 3-Channel Gate

Wire plate, depth and normals to the three channels to switch an entire reference set together.

#### 4-Channel Gate

Wire image, prompt, model name and seed separately. One switch releases the set for export.

#### Enumerate List

Number takes by connecting their list. Three takes yield the index list `0, 1, 2`.

#### Get List Length

Count variants with `length`. It counts ComfyUI list items. To count frames inside a batch this way, use Create Image List first.

#### Edit List

Fit references to a model limit. `edit_length = -1` keeps the length, `0` clears the list. Asking `Remove_Spaced` for three of `A B C D E` keeps `A C E`. To extend, `Loop` cycles the input. `Repeat_First` and `Repeat_Last` append copies of the chosen end item. The default `No_lengthening` leaves short lists alone. Seeded `shuffle` runs after resizing.

#### Append to List

Place the hero reference at `First`, `Last` or `At Position`. Positions start at zero. An index past the end appends. An inserted list stays one item.

### JSON Tools

Read shot briefs in the graph and assemble API payloads from their fields. When building a dict from JSON or a key–value pair, **Strip new Lines** replaces line breaks inside string values with spaces, even in nested fields.

#### Generate DICT from JSON (JSON-Tools)

Feed in a JSON object such as `{"shots":[]}`, with or without Markdown code fences. Wrap a top-level list in a named field such as `shots`.

#### Generate DICT from Key-Value-Pair (JSON-Tools)

Key `takes` and value `3` give `{"takes":3}`. JSON values and fragments are parsed. Plain text stays text. An empty key or value yields an empty dict.

#### Generate Dict of Dicts (JSON-Tools)

Group fields under a shot ID. With `key` set to `sh007`, `{"stage":"plate"}` and `{"takes":3}` become `{"sh007":{"stage":"plate","takes":3}}`. Later inputs win on duplicate keys.

#### Entry in DICT (JSON-Tools)

Check `shots[0].dialogue` before enabling a dialogue branch. `Exists` is true whenever the path exists, even with an empty value.

#### Get Value from DICT (JSON-Tools)

Read a field by path. For `shots[0].refs`, `As Dict` preserves the value under `refs`. An array gives JSON text on `As String` and one string per item on `As String List`. A missing path raises an error.

#### Get All Values from DICT (JSON-Tools)

Collect top-level values for iteration. `{"takes":3,"refs":["A","B"]}` yields `[3]` by default or `[3,["A","B"]]` with `Include Nested`. The second output gives the selected key–value pairs as compact JSON.

#### Generate JSON from DICT (JSON-Tools)

Connect the finished dict. Send compact `JSON` to the API. `Preview` indents it and expands embedded newlines for reading.

### String Tools

Build prompts and filenames from graph values, or pull one field out of a structured name.

#### FString Input

Use placeholders in a prompt or filename. `sh{input_0:03d}` with integer `7` on `input_0` yields `sh007`. Add `{input_1}` for another value, or use `{{` and `}}` for literal braces.

#### Join Strings

Connect name components in order. Joining `sh007` and `plate` with delimiter `_` gives `sh007_plate`. An empty delimiter joins them directly.

#### String Split at Symbol

Split `sh007_plate_v2` on `_` and use index `1` to get `plate`. Indices start at zero. An out-of-range index returns an empty string.

#### Any to String

Turn a value into text. Lists and dicts use JSON where possible. UTF-8 bytes are decoded as text.

### Async Tools

Stagger requests when several branches would otherwise hit an API at once.

#### Async Delay

Route a value the call depends on through this node. With `delay` set to `1000` milliseconds, it forwards the same value one second later while other branches can run.
