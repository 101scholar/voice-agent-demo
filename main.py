from utils import get_voice_input, play_audio
import asyncio
import numpy as np
from agents.voice import VoicePipeline, VoiceWorkflowBase, STTModel, TTSModel, StreamedTranscriptionSession
from agents import trace, function_tool, Runner, Agent
from openai.types.responses import ResponseTextDeltaEvent
import sherpa_onnx
from utils_live2d import live2d_message, live2d_voice

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()


class Workflow(VoiceWorkflowBase):
    def __init__(self):
        self.agent = Agent(
            name="Chat Agent",
            instructions="Use a conversational tone and write in a chat style without formal formatting or lists and do not use any emojis. Directly output with no explanation.",
            model="litellm/xai/grok-3-beta",
        )
        self.history = []

        def clear_history():
            """ Clear Agent Historical conversation
            """
            self.history = None
        
        self.agent.tools.append(function_tool(clear_history))

    async def run(self, transcription):
        self.history.append({
            "content": transcription,
            "role": "user"
        })
        output = Runner.run_streamed(
            self.agent,
            self.history,
        )
        content = ""
        async for event in output.stream_events():
            if event.type == "raw_response_event" and isinstance(
                    event.data, ResponseTextDeltaEvent
                ):
                content = f"{content}{event.data.delta}"
                live2d_message(content)
                yield event.data.delta
        if self.history is None:
            self.history = []
        else:
            self.history = output.to_input_list()


class SherpaOnnxSTTTranscriptionSession(StreamedTranscriptionSession):
    async def transcribe_turns(self):
        yield ""

    async def close(self):
        pass


class SherpaOnnxSTTModel(STTModel):
    """A speech-to-text model for OpenAI."""

    def __init__(self):
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model="./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.int8.onnx",
            tokens="./models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
            num_threads=4,
            use_itn=True,
            debug=False,
            provider="cpu",
        )

    @property
    def model_name(self):
        return "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

    async def transcribe(
        self,
        input,
        settings,
        trace_include_sensitive_data,
        trace_include_sensitive_audio_data,
    ):
        """Transcribe an audio input.

        Args:
            input: The audio input to transcribe.
            settings: The settings to use for the transcription.

        Returns:
            The transcribed text.
        """
        stream = self.recognizer.create_stream()
        stream.accept_waveform(input.frame_rate, input.buffer)
        self.recognizer.decode_streams([stream])
        return stream.result.text

    async def create_session(
        self,
        input,
        settings,
        trace_include_sensitive_data,
        trace_include_sensitive_audio_data,
    ):
        return SherpaOnnxSTTTranscriptionSession()


class SherpaOnnxTTSModel(TTSModel):
    """A text-to-speech local model."""

    def __init__(self):
        self.tts = sherpa_onnx.OfflineTts(
            sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model="./models/kokoro-multi-lang-v1_0/model.onnx",
                        lexicon="./models/kokoro-multi-lang-v1_0/lexicon-us-en.txt,./models/kokoro-multi-lang-v1_0/lexicon-zh.txt",
                        voices="./models/kokoro-multi-lang-v1_0/voices.bin",
                        length_scale=1.,
                        data_dir="./models/kokoro-multi-lang-v1_0/espeak-ng-data",
                        dict_dir="./models/kokoro-multi-lang-v1_0/dict",
                        tokens="./models/kokoro-multi-lang-v1_0/tokens.txt",
                    ),
                    provider="cpu",
                    debug=False,
                    num_threads=1,
                ),
                rule_fsts="./models/kokoro-multi-lang-v1_0/date-zh.fst,./models/kokoro-multi-lang-v1_0/phone-zh.fst,./models/kokoro-multi-lang-v1_0/number-zh.fst",
                max_num_sentences=1,
            )
        )
        self.sample_rate = 24000

    @property
    def model_name(self):
        return "SherpaOnnx-TTS"

    async def run(self, text, settings):
        """Run the text-to-speech model.

        Args:
            text: The text to convert to speech.
            settings: The settings to use for the text-to-speech model.

        Returns:
            An iterator of audio chunks.
        """
        audio = self.tts.generate(text, sid=47, speed=1.0)
        yield (np.array(audio.samples) * 32767).astype(np.int16).tobytes()


async def main():
    workflow = Workflow()
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=SherpaOnnxSTTModel(),
        tts_model=SherpaOnnxTTSModel(),
    )

    with trace("Voice Agent Chat"):
        audio_input = get_voice_input(
            lambda: live2d_message("发呆中"),
            lambda: live2d_message("我在听"),
        )
        print("处理中...")
        result = await pipeline.run(audio_input)

        chunks = []
        async for event in result.stream():
            if event.type == "voice_stream_event_audio":
                chunks.append(event.data)
        audio = np.concatenate(chunks, axis=0)
        live2d_voice(audio)



if __name__ == "__main__":
    asyncio.run(main())
