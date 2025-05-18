import sounddevice as sd
import numpy as np

from enum import Enum
from agents.voice import AudioInput


def play_audio(audio, samplerate=24000):
    sd.play(
        audio,
        samplerate=samplerate
    )
    sd.wait()


def is_silence(audio_chunk, threshold=3000):
    # 检测音频片段是否为静音
    # print(np.max(np.abs(audio_chunk)))
    return np.max(np.abs(audio_chunk)) < threshold


class MicState(Enum):
    INIT = 0
    IDLE = 1
    RECORDING = 2
    STOPPING = 3


def get_voice_input(idle_callback=None, recording_callback=None):
    samplerate = 16000
    recorded_chunks = []

    max_silence_duration = 2.0   # 静音持续时间阈值(秒)
    chunk_duration = 0.1     # 每个音频块的持续时间(秒)
    chunk_size = int(samplerate * chunk_duration)

    mic_state = MicState.INIT
    silence_duration = 0.
    def audio_callback(indata, frames, time_info, status):
        nonlocal mic_state
        if mic_state == MicState.RECORDING:
            nonlocal silence_duration
            if is_silence(indata):
                silence_duration += chunk_duration
            else:
                silence_duration = 0.
                recorded_chunks.append(indata.copy())
        elif mic_state == MicState.IDLE:
            if not is_silence(indata):
                if recording_callback:
                    recording_callback()
                mic_state = MicState.RECORDING
                recorded_chunks.append(indata.copy())
        elif mic_state == MicState.INIT:
            if idle_callback:
                idle_callback()
            if not is_silence(indata):
                if recording_callback:
                    recording_callback()
                mic_state = MicState.RECORDING
                recorded_chunks.append(indata.copy())
            else:
                mic_state = MicState.IDLE

    with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype='int16',
            blocksize=chunk_size,
            callback=audio_callback
        ):
        while True:
            sd.sleep(int(0.1 * 1000))
            if silence_duration >= max_silence_duration:
                mic_state = MicState.STOPPING
                break

    recording = np.concatenate(recorded_chunks, axis=0)

    audio_input = AudioInput(
        buffer=recording,
        frame_rate=samplerate
    )
    return audio_input
