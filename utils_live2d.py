import json
from websockets.sync.client import connect
import base64
from time import sleep
from pydub import AudioSegment
from io import BytesIO

def live2d_send_json(data):
    with connect("ws://localhost:10086/api") as websocket:
        websocket.send(json.dumps(data))


def live2d_message(content: str, duration: int = -1):
    data = {
        "msg": 11000,
        "msgId": 1,
        "data": {
            "id": 0,
            "text": content,
            "textFrameColor": 0x000000,
            "textColor": 0xFFFFFF,
            "duration": duration
        }
    }
    live2d_send_json(data)


def _live2d_voice(mp3_bytes, duration):
    sound = base64.b64encode(mp3_bytes).decode('utf-8')
    data = {
        "msg": 13500,
        "msgId": 1,
        "data": {
            "id": 0,
            "channel":0,
            "volume":1,
            "delay":0,
            "loop": False,
            "type": 1,
            "sound": sound
        }
    }
    live2d_send_json(data)
    sleep(duration)


def audio_to_mp3bytes(audio, frame_rate=24000):
    bytes_io = BytesIO()
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=frame_rate,
        sample_width=2,
        channels=1
    )
    audio_segment.export(bytes_io, format="mp3")
    mp3_bytes = bytes_io.getvalue()

    return mp3_bytes


def get_audio_duration(audio, frame_rate=24000):
    duration = len(audio) / frame_rate
    return duration


def live2d_voice(audio, frame_rate=24000):
    mp3 = audio_to_mp3bytes(audio, frame_rate)
    duration = get_audio_duration(audio, frame_rate)
    _live2d_voice(mp3, duration)

