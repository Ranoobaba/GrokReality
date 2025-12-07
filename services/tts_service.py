"""
Text-to-speech service using Grok Voice API.
"""
import os
import base64
import requests
from typing import Optional
from config import XAI_API_KEY, TTS_ENDPOINT, MAX_INPUT_LENGTH


def file_to_base64(file_path: str) -> str:
    """Convert a file to base64 string."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def tts_request(input_text: str, voice_file: Optional[str] = None) -> Optional[bytes]:
    """Generate speech from text using the TTS API."""
    if not XAI_API_KEY:
        return None

    if voice_file and os.path.exists(voice_file):
        voice_base64 = file_to_base64(voice_file)
    else:
        voice_base64 = None

    input_text = input_text[:MAX_INPUT_LENGTH]

    payload = {
        "model": "grok-voice",
        "input": input_text,
        "response_format": "mp3",
        "instructions": "audio",
        "voice": voice_base64 or "None",
        "sampling_params": {
            "max_new_tokens": 512,
            "temperature": 1.0,
            "min_p": 0.01,
        },
    }

    response = requests.post(
        TTS_ENDPOINT,
        json=payload,
        stream=True,
        headers={"Authorization": f"Bearer {XAI_API_KEY}"}
    )

    if response.status_code == 200:
        return response.content
    return None

