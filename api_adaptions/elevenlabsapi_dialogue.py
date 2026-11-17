"""Text to Dialogue with a JSON turn list and graph-selected voices."""

import json

from comfy_api.latest import IO
from comfy_api_nodes.nodes_elevenlabs import ElevenLabsTextToDialogue


class KASKIElevenLabsJSONToDialogue(ElevenLabsTextToDialogue):
    @classmethod
    def define_schema(cls) -> IO.Schema:
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
                IO.String.Input(
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
    async def execute(cls, inputs: dict, **settings) -> IO.NodeOutput:
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
