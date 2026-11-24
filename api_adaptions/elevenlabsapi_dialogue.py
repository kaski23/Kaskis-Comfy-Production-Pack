"""Text to Dialogue with a JSON turn list and graph-selected voices."""

import json

from comfy_api.latest import io
from comfy_api_nodes.nodes_elevenlabs import (
    ElevenLabsTextToDialogue,
    ElevenLabsVoiceSelector,
)


# Local voice catalogue. Edit this list independently of ComfyUI.
# (voice_id, display_name, gender, accent)
ELEVENLABS_VOICES: list[tuple[str, str, str, str]] = [
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "male", "american"),
    ("EXAVITQu4vr4xnSDxMaL", "Sarah", "female", "american"),
    ("FGY2WhTYpPnrIDTdsKH5", "Laura", "female", "american"),
    ("IKne3meq5aSn9XLyUdCD", "Charlie", "male", "australian"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "male", "british"),
    ("N2lVS1w4EtoT3dr4eOWO", "Callum", "male", "american"),
    ("SAz9YHcvj6GT2YYXdXww", "River", "neutral", "american"),
    ("SOYHLrjzK2X1ezoPC6cr", "Harry", "male", "american"),
    ("TX3LPaxmHKxFdv7VOQHJ", "Liam", "male", "american"),
    ("Xb7hH8MSUJpSbSDYk0k2", "Alice", "female", "british"),
    ("XrExE9yKIg1WjnnlVkGX", "Matilda", "female", "american"),
    ("bIHbv24MWmeRgasZH58o", "Will", "male", "american"),
    ("cgSgspJ2msm6clMCkdW9", "Jessica", "female", "american"),
    ("cjVigY5qzO86Huf0OWal", "Eric", "male", "american"),
    ("hpp4J3VqNfWAUOO0d1Us", "Bella", "female", "american"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "male", "american"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "male", "american"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "male", "british"),
    ("pFZP5JQG7iQjIQuC4Bku", "Lily", "female", "british"),
    ("pNInz6obpgDQGcFmaJgB", "Adam", "male", "american"),
    ("pqHfZKP75CvOlQylNhV4", "Bill", "male", "american"),
]


class KASKIElevenLabsVoiceSelector(ElevenLabsVoiceSelector):
    @classmethod
    def _voice_map(cls) -> dict[str, str]:
        return {
            f"{name} ({gender}, {accent})": voice_id
            for voice_id, name, gender, accent in ELEVENLABS_VOICES
        }

    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ElevenLabsVoiceSelector_KASKI"
        schema.display_name = "ElevenLabs Voice Selector (KASKI)"
        schema.description = (
            "Select a voice from this file's local catalogue, "
            "or override the selection with alternative_voice_id."
        )
        schema.category = "KASKI/api-adaptions/audio"

        voice_options = next(widget for widget in schema.inputs if widget.id == "voice")
        voice_options.options = list(cls._voice_map())
        voice_options.default = voice_options.options[0] if voice_options.options else None
        voice_options.tooltip = "Choose a voice from this file's local voice lists."
        schema.inputs.append(
            io.String.Input(
                "alternative_voice_id",
                default="",
                dynamic_prompts=False,
                optional=True,
                tooltip=(
                    "Alternative manual ElevenLabs voice ID. "
                    "A non-empty ID overrides the voice dropdown. "
                    "Leave empty to use the selected voice."
                ),
            )
        )
        return schema

    @classmethod
    def execute(cls, voice: str, alternative_voice_id: str = "") -> io.NodeOutput:
        manual_voice_id = alternative_voice_id.strip()
        if manual_voice_id:
            return io.NodeOutput(manual_voice_id)

        selected_voice_id = cls._voice_map().get(voice)
        if not selected_voice_id:
            raise ValueError(f"Unknown voice: {voice}")
        return io.NodeOutput(selected_voice_id)


class KASKIElevenLabsJSONToDialogue(ElevenLabsTextToDialogue):
    @classmethod
    def define_schema(cls) -> io.Schema:
        schema = super().define_schema()
        schema.node_id = "ElevenLabsJSONToDialogue_KASKI"
        schema.display_name = "ElevenLabs JSON to Dialogue (KASKI)"
        schema.description = "Generate dialogue from a JSON array of text and voice_id pairs."
        schema.category = "KASKI/api-adaptions/audio"

        voice_options = next(widget for widget in schema.inputs if widget.id == "inputs")
        voice_options.tooltip = "Number of voice sockets to expose. Connect the voices referenced in the JSON."
        for option in voice_options.options:
            option.inputs = [socket for socket in option.inputs if socket.id.startswith("voice")]
            for socket in option.inputs:
                socket.optional = True
                socket.tooltip = (
                    f'Use "{socket.id}" as voice_id for any turn that should use this voice. '
                    "Connect a Voice Selector or Instant Voice Clone."
                )

            # Mount the JSON widget together with the voice sockets.
            option.inputs.insert(
                0,
                io.String.Input(
                    "dialogue_json",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip=(
                        'A JSON array of dialogue turns. Use socket names such as "voice1" as voice_id.\n\n'
                        '[{"text": "[solemn] Willkommen.", "voice_id": "voice1"}, '
                        '{"text": "Danke.", "voice_id": "voice2"}]'
                    ),
                ),
            )
        return schema

    @classmethod
    async def execute(cls, inputs: dict, **settings) -> io.NodeOutput:
        turns = json.loads(inputs["dialogue_json"])
        if not isinstance(turns, list) or not turns:
            raise ValueError("dialogue_json must contain a non-empty JSON array.")

        voices = {name: value for name, value in inputs.items() if name.startswith("voice")}
        core_inputs: dict[str, int | str] = {"inputs": len(turns)}
        for number, turn in enumerate(turns, start=1):
            if not isinstance(turn, dict) or not isinstance(turn.get("text"), str):
                raise ValueError(f"Dialogue turn {number} needs a 'text' string.")

            socket = turn.get("voice_id")
            voice_id = voices.get(socket) if isinstance(socket, str) else None
            if not isinstance(voice_id, str) or not voice_id.strip():
                raise ValueError(
                    f"Dialogue turn {number} references {socket!r}. Select and connect the matching voice socket."
                )

            core_inputs[f"text{number}"] = turn["text"]
            core_inputs[f"voice{number}"] = voice_id

        return await super().execute(inputs=core_inputs, **settings)

ELEVENLABSAPI_NODES_LIST = [
    KASKIElevenLabsVoiceSelector,
    KASKIElevenLabsJSONToDialogue,
]