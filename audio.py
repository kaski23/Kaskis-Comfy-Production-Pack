
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
        

import torch
import torch.nn.functional as F

from comfy_api.latest import ComfyExtension, io


class AudioResampler_KASKI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioResampler_KASKI",
            display_name="Audio Resampler",
            category="audio",
            description=(
                "Resampelt Audio auf die Zielrate. Ausgabe immer als float32. "
                "Verwendet wie die Originalversion lineare Interpolation "
                "ohne Anti-Aliasing-Filter."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip=(
                        "Eingangsaudio mit Waveform [Batch, Kanäle, Samples] "
                        "und sample_rate in Hz."
                    ),
                ),
                io.Int.Input(
                    "override_input_samplerate_khz",
                    default=-1,
                    min=-1,
                    max=192000,
                    tooltip=(
                        "-1 deaktiviert den Override und verwendet die "
                        "Abtastrate des Eingangsaudios. Positive Werte "
                        "interpretieren die vorhandenen Samples vor dem "
                        "Resampling mit dieser Quellrate in kHz "
                        "(48 = 48000 Hz). Eine abweichende Quellrate "
                        "verändert Dauer und Tonhöhe. 0 ist ungültig."
                    ),
                ),
                io.Int.Input(
                    "output_samplerate_khz",
                    default=48,
                    min=1,
                    max=192000,
                    tooltip=(
                        "Ziel-Abtastrate in kHz (48 = 48000 Hz). Resampling "
                        "erfolgt automatisch, wenn Zielrate und effektive "
                        "Quellrate verschieden sind. Ohne Override bleiben "
                        "Dauer und Tonhöhe bis auf Sample-Rundung erhalten."
                    ),
                ),
            ],
            outputs=[
                io.Audio.Output(
                    display_name="audio",
                    tooltip="Audio mit float32-Waveform und der Ziel-Abtastrate in Hz.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        audio: io.Audio.Type,
        override_input_samplerate_khz: int = -1,
        output_samplerate_khz: int = 48,
    ) -> io.NodeOutput:
        if override_input_samplerate_khz != -1 and override_input_samplerate_khz <= 0:
            raise ValueError("Der Eingangs-Override muss -1 oder größer als 0 sein.")

        samplerate_in = (
            int(audio["sample_rate"])
            if override_input_samplerate_khz == -1
            else override_input_samplerate_khz * 1000
        )
        samplerate_out = output_samplerate_khz * 1000

        if samplerate_in <= 0 or samplerate_out <= 0:
            raise ValueError("Quell- und Ziel-Abtastrate müssen größer als 0 sein.")

        waveform: torch.Tensor = audio["waveform"]
        if waveform.ndim != 3 or waveform.shape[-1] == 0:
            raise ValueError("Erwartet eine nicht leere Waveform [Batch, Kanäle, Samples].")

        # Preserve the original PCM normalization; output is always float32.
        pcm_divisors = {
            torch.int8: 128.0,
            torch.int16: 32768.0,
            torch.int32: 2147483648.0,
            torch.int64: 9223372036854775808.0,
        }
        if waveform.dtype in pcm_divisors:
            waveform = waveform.to(torch.float32) / pcm_divisors[waveform.dtype]
        elif waveform.dtype.is_floating_point:
            waveform = waveform.to(torch.float32).clamp(-1.0, 1.0)
        else:
            waveform = waveform.to(torch.float32)

        if samplerate_out != samplerate_in:
            batch, channels, samples = waveform.shape
            new_samples = max(1, int(round(samples * samplerate_out / samplerate_in)))
            waveform = F.interpolate(
                waveform.reshape(batch * channels, 1, samples),
                size=new_samples,
                mode="linear",
                align_corners=False,
            ).reshape(batch, channels, new_samples)

        return io.NodeOutput({
            **audio,
            "waveform": waveform,
            "sample_rate": samplerate_out,
        })

        
        
AUDIO_NODES_LIST = [
    GetAudioLength_KASKI,
    AudioResampler_KASKI,
]
