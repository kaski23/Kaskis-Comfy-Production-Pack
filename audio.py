
from comfy_api.latest import io


class GetAudioLength_KASKI(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GetAudioLength_KASKI",
            display_name="Get Audio Length",
            category="KASKI/audio",
            description="Returns the duration of a ComfyUI AUDIO object in seconds.",
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="Audio to measure. Uses sample count and sample rate.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="length_seconds"),
            ],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        sample_rate = audio["sample_rate"]
        if sample_rate <= 0:
            raise ValueError("Audio sample_rate must be greater than zero.")

        seconds = audio["waveform"].shape[-1] / sample_rate
        return io.NodeOutput(float(seconds))
        
        
AUDIO_NODES_LIST = [
    GetAudioLength_KASKI,
]
