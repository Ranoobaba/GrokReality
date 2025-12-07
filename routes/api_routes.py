"""
API routes for the FastAPI application.
"""

import asyncio
import base64
import json
import os
import queue
import shutil
import tempfile
import threading
from typing import Optional

import speech_recognition as sr
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse

from config import ASSEMBLYAI_API_KEY, UPLOAD_FOLDER
from models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    PodcastData,
    TranscribeResponse,
    UploadPodcastResponse,
    Utterance,
)
from services.audio_service import (
    convert_audio_to_wav,
    download_audio_from_url,
    extract_voice_sample_from_podcast,
)
from services.grok_service import call_grok_llm
from services.transcription_service import transcribe_podcast_with_assemblyai
from services.tts_service import tts_request
from services.x_service import search_x_tweets

router = APIRouter(prefix="/api", tags=["api"])

# Store podcast data in memory (in production, use a database)
podcast_data: dict[str, PodcastData] = {}


@router.post(
    "/transcribe",
    response_model=TranscribeResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def transcribe(
    audio: UploadFile = File(..., description="Audio file to transcribe"),
):
    """Transcribe user voice input to text."""
    try:
        # Save to temporary file (keep original extension)
        suffix = ".webm"
        if audio.filename:
            ext = os.path.splitext(audio.filename)[1]
            if ext:
                suffix = ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        wav_path = None
        try:
            # Convert to WAV format for speech recognition
            wav_path = convert_audio_to_wav(tmp_path)
            if not wav_path or not os.path.exists(wav_path):
                # If conversion failed, try original file
                wav_path = tmp_path

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data)
            return TranscribeResponse(text=text)

        finally:
            # Clean up temp files
            for path in [
                tmp_path,
                wav_path if wav_path and wav_path != tmp_path else None,
            ]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass

    except sr.UnknownValueError:
        raise HTTPException(status_code=400, detail="Could not understand audio")
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/upload-podcast",
    response_model=UploadPodcastResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def upload_podcast(
    speaker_name: str = Form(..., description="Name of the podcast speaker"),
    podcast_url: Optional[str] = Form(default=None, description="URL to podcast audio"),
    podcast: Optional[UploadFile] = File(
        default=None, description="Podcast audio file"
    ),
):
    """Upload and transcribe podcast audio from file or URL."""
    speaker_name = speaker_name.strip()
    if not speaker_name:
        raise HTTPException(status_code=400, detail="Speaker name is required")

    podcast_url = podcast_url.strip() if podcast_url else None

    filepath = None
    cleanup_file = False

    try:
        # Handle URL or file upload
        if podcast_url:
            # Download from URL
            print(f"Downloading podcast from URL: {podcast_url}")
            filepath = download_audio_from_url(podcast_url)
            if not filepath:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to download audio from URL. Check server logs for details.",
                )
            cleanup_file = True
            # Use URL hash for unique ID
            podcast_id = f"url_{abs(hash(podcast_url)) % 100000}"
        elif podcast and podcast.filename:
            # Handle file upload
            filename = os.path.basename(podcast.filename)
            # Sanitize filename
            filename = "".join(c for c in filename if c.isalnum() or c in "._-")
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Ensure upload directory exists
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)

            # Save uploaded file
            with open(filepath, "wb") as buffer:
                content = await podcast.read()
                buffer.write(content)
            podcast_id = filename
        else:
            raise HTTPException(
                status_code=400, detail="Either podcast file or URL is required"
            )

        if not filepath or not os.path.exists(filepath):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get audio file. Filepath: {filepath}",
            )

        # Verify file is readable and has content
        file_size = os.path.getsize(filepath)
        print(f"Audio file ready: {filepath} ({file_size} bytes)")

        if file_size == 0:
            raise HTTPException(status_code=500, detail="Downloaded file is empty")

        # Transcribe with AssemblyAI
        print(f"Starting transcription of {filepath}...")
        transcript_data = transcribe_podcast_with_assemblyai(filepath)

        if not transcript_data:
            raise HTTPException(status_code=500, detail="Failed to transcribe podcast")

        # Extract transcript text and speaker information
        transcript_text = transcript_data.get("text", "")
        utterances_raw = transcript_data.get("utterances", [])

        # Format transcript with speaker labels
        formatted_transcript = ""
        utterances = []
        for utterance in utterances_raw:
            speaker = utterance.get("speaker", "Unknown")
            text = utterance.get("text", "")
            formatted_transcript += f"Speaker {speaker}: {text}\n"
            utterances.append(
                Utterance(
                    speaker=speaker,
                    text=text,
                    start=utterance.get("start"),
                    end=utterance.get("end"),
                )
            )

        # Extract voice sample from podcast for voice cloning
        voice_sample_path = None
        if filepath and os.path.exists(filepath):
            print("Extracting voice sample from podcast...")
            voice_sample_path = extract_voice_sample_from_podcast(
                filepath, duration_seconds=30
            )

        # Store podcast data
        podcast_data[podcast_id] = PodcastData(
            speaker_name=speaker_name,
            transcript=transcript_text,
            formatted_transcript=formatted_transcript,
            utterances=utterances,
            podcast_file_path=filepath,
            voice_sample_path=voice_sample_path,
        )

        # IMPORTANT: Keep the podcast file for voice cloning!
        # Don't delete it - we need it to generate responses in the podcast host's voice
        if cleanup_file:
            # For URL downloads, move to uploads folder so it persists for voice cloning
            try:
                os.makedirs(UPLOAD_FOLDER, exist_ok=True)
                filename = os.path.basename(filepath) or f"podcast_{podcast_id}.mp3"
                persistent_path = os.path.join(UPLOAD_FOLDER, filename)

                # Only move if not already in uploads folder
                if not filepath.startswith(UPLOAD_FOLDER):
                    if os.path.exists(filepath):
                        shutil.move(filepath, persistent_path)
                        filepath = persistent_path
                        # Update stored path
                        podcast_data[podcast_id].podcast_file_path = filepath
                        print(
                            f"✅ Moved podcast file to persistent location: {filepath}"
                        )
            except Exception as e:
                print(f"Error moving podcast file (keeping original): {e}")

        transcript_preview = (
            transcript_text[:500] + "..."
            if len(transcript_text) > 500
            else transcript_text
        )

        return UploadPodcastResponse(
            podcast_id=podcast_id,
            speaker_name=speaker_name,
            transcript_preview=transcript_preview,
            message="Podcast transcribed successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {e}")
        # Clean up on error
        if cleanup_file and filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat(request: ChatRequest):
    """Handle chat request: send to Grok with podcast context and return audio response."""
    try:
        user_input = request.text
        podcast_id = request.podcast_id

        print(
            f"📨 Chat request - User input: {user_input[:50]}..., Podcast ID: {podcast_id}"
        )

        if not user_input:
            raise HTTPException(status_code=400, detail="No text provided")

        # Get podcast context
        podcast_transcript = ""
        speaker_name = ""
        if podcast_id:
            if podcast_id in podcast_data:
                podcast_info = podcast_data[podcast_id]
                podcast_transcript = podcast_info.formatted_transcript
                speaker_name = podcast_info.speaker_name
                print(
                    f"📻 Found podcast context - Speaker: {speaker_name}, Transcript length: {len(podcast_transcript)}"
                )
            else:
                print(
                    f"⚠️  Podcast ID {podcast_id} not found in podcast_data. Available IDs: {list(podcast_data.keys())}"
                )

        # Search X for speaker tweets
        x_tweets = []
        if speaker_name:
            # Try different username variations
            base_username = speaker_name.lower().replace(" ", "").replace("-", "")
            username_variations = [
                base_username,
                f"{base_username}official",
                f"{base_username}podcast",
                f"{base_username}show",
            ]

            # Try each variation until we find tweets
            for username in username_variations:
                x_tweets = search_x_tweets(
                    username, query=user_input[:50], max_results=5
                )
                if x_tweets:
                    print(f"🐦 Found {len(x_tweets)} tweets from @{username}")
                    break

        # Call Grok LLM with context
        print("🤖 Calling Grok LLM...")
        grok_response = call_grok_llm(
            user_input,
            podcast_transcript=podcast_transcript,
            speaker_name=speaker_name,
            x_tweets=x_tweets,
        )

        if not grok_response:
            raise HTTPException(
                status_code=500, detail="Failed to get response from Grok"
            )

        print(f"✅ Grok response received: {grok_response[:100]}...")

        # Use the podcast audio file as the voice file (like demo.py pattern)
        voice_file = None

        # If podcast_id provided but not found, try to use Joe Rogan voice as fallback
        if podcast_id and podcast_id not in podcast_data:
            print(
                f"⚠️  Podcast ID {podcast_id} not found. Using Joe Rogan voice as fallback."
            )
            if not speaker_name:
                speaker_name = "Joe Rogan"

        if podcast_id and podcast_id in podcast_data:
            podcast_info = podcast_data[podcast_id]

            # Priority 1: Use the original podcast file (best quality, full voice)
            podcast_file_path = podcast_info.podcast_file_path
            if podcast_file_path and os.path.exists(podcast_file_path):
                voice_file = podcast_file_path
                print(f"✅ Using podcast audio file for voice cloning: {voice_file}")
            # Priority 2: Use extracted voice sample if original not available
            else:
                voice_sample = podcast_info.voice_sample_path
                if voice_sample and os.path.exists(voice_sample):
                    voice_file = voice_sample
                    print(f"✅ Using podcast voice sample: {voice_sample}")

        # Fallback: Try voices directory (especially joe-rogan.mp3)
        if not voice_file:
            voices_dir = os.path.join(
                os.path.dirname(__file__), "..", "voice-demo-hackathon 2", "voices"
            )
            if os.path.exists(voices_dir):
                # Try to find a voice file matching the speaker name
                if speaker_name:
                    speaker_lower = speaker_name.lower().replace(" ", "-")
                    name_variations = [
                        speaker_lower,
                        speaker_name.lower().replace(" ", ""),
                        speaker_name.lower().replace(" ", "_"),
                    ]

                    for name_var in name_variations:
                        possible_files = [
                            f"{name_var}.m4a",
                            f"{name_var}.mp3",
                            f"{name_var}.wav",
                        ]
                        for filename in possible_files:
                            file_path = os.path.join(voices_dir, filename)
                            if os.path.exists(file_path):
                                voice_file = file_path
                                print(
                                    f"✅ Using voice file from voices directory: {voice_file}"
                                )
                                break
                        if voice_file:
                            break

                # Fallback: Try joe-rogan.mp3 specifically (common case)
                if not voice_file:
                    joe_rogan_file = os.path.join(voices_dir, "joe-rogan.mp3")
                    if os.path.exists(joe_rogan_file):
                        voice_file = joe_rogan_file
                        print(f"✅ Using Joe Rogan voice file: {voice_file}")

                # Final fallback to any available voice file
                if not voice_file:
                    voice_files = [
                        f
                        for f in os.listdir(voices_dir)
                        if f.endswith((".m4a", ".mp3", ".wav"))
                    ]
                    if voice_files:
                        voice_file = os.path.join(voices_dir, voice_files[0])
                        print(f"Using fallback voice file: {voice_file}")

        if not voice_file:
            print("⚠️  Warning: No voice file found, TTS will use default voice")
        else:
            print(
                f"🎤 Generating response with voice from: {os.path.basename(voice_file)}"
            )

        print("🎵 Generating TTS audio...")
        audio_data = tts_request(grok_response, voice_file)

        if not audio_data:
            raise HTTPException(status_code=500, detail="Failed to generate audio")

        print(f"✅ Audio generated: {len(audio_data)} bytes")

        # Return audio as base64
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        return ChatResponse(text=grok_response, audio=audio_base64)

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


class WebSocketAudioStream:
    """Custom audio stream that receives data from a queue for AssemblyAI streaming."""

    def __init__(self):
        self.queue = queue.Queue()
        self.running = True

    def write(self, data: bytes):
        """Add audio data to the queue."""
        if self.running:
            self.queue.put(data)

    def stop(self):
        """Signal to stop the stream."""
        self.running = False
        self.queue.put(None)  # Sentinel to unblock the iterator

    def __iter__(self):
        return self

    def __next__(self):
        if not self.running:
            raise StopIteration
        data = self.queue.get()
        if data is None:
            raise StopIteration
        return data


@router.websocket("/stream-transcribe")
async def stream_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transcription with turn detection.

    Client sends audio chunks, server streams back transcription.
    Uses AssemblyAI's Universal-2 Streaming API with automatic turn detection.
    """
    await websocket.accept()
    print("🎤 WebSocket connection established for streaming transcription")

    if not ASSEMBLYAI_API_KEY:
        await websocket.send_json(
            {"type": "error", "message": "AssemblyAI API key not configured"}
        )
        await websocket.close()
        return

    try:
        import assemblyai as aai
        from assemblyai.streaming.v3 import (
            BeginEvent,
            StreamingClient,
            StreamingClientOptions,
            StreamingError,
            StreamingEvents,
            StreamingParameters,
            TerminationEvent,
            TurnEvent,
        )
    except ImportError as e:
        print(f"❌ AssemblyAI import error: {e}")
        await websocket.send_json(
            {
                "type": "error",
                "message": "AssemblyAI SDK not installed or outdated. Run: pip install -U assemblyai",
            }
        )
        await websocket.close()
        return

    # Variables to track state
    final_transcript = ""
    client = None
    is_connected = True
    transcript_ready = asyncio.Event()
    loop = asyncio.get_event_loop()
    audio_stream = None
    stream_thread = None

    # Async-safe function to send messages to websocket from sync callbacks
    def send_ws_message(msg_type: str, **kwargs):
        if is_connected:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": msg_type, **kwargs}), loop
                )
                future.result(timeout=1.0)
            except Exception as e:
                print(f"⚠️ Failed to send WS message: {e}")

    def set_transcript_ready():
        loop.call_soon_threadsafe(transcript_ready.set)

    try:
        # Create streaming client with new Universal-2 API
        client = StreamingClient(
            StreamingClientOptions(
                api_key=ASSEMBLYAI_API_KEY,
            )
        )

        # Event handlers
        def on_begin(self, event: BeginEvent):
            print(f"🎙️ AssemblyAI session started: {event.id}")

        def on_turn(self, event: TurnEvent):
            nonlocal final_transcript
            if not is_connected:
                return

            # Send partial transcript for live feedback
            if event.transcript:
                send_ws_message("partial", text=event.transcript)

            # Check if this is the end of the user's turn (they stopped speaking)
            # Only process if we have actual text and haven't already detected end of turn
            if event.end_of_turn and event.transcript and event.transcript.strip():
                # Only set if we don't already have a final transcript (prevent race condition)
                if not final_transcript:
                    final_transcript = event.transcript
                    print(f"✅ End of turn detected: {final_transcript}")
                    set_transcript_ready()
                else:
                    print(f"⚠️ Ignoring duplicate end_of_turn (already have transcript)")

        def on_terminated(self, event: TerminationEvent):
            print(
                f"🔒 AssemblyAI session terminated: {event.audio_duration_seconds}s processed"
            )

        def on_error(self, error: StreamingError):
            print(f"❌ AssemblyAI streaming error: {error}")
            send_ws_message("error", message=str(error))

        # Register event handlers
        client.on(StreamingEvents.Begin, on_begin)
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Termination, on_terminated)
        client.on(StreamingEvents.Error, on_error)

        # Connect to AssemblyAI
        client.connect(
            StreamingParameters(
                sample_rate=16000,
            )
        )

        # Create audio stream and start streaming in background thread
        audio_stream = WebSocketAudioStream()

        def run_stream():
            try:
                client.stream(audio_stream)
            except Exception as e:
                print(f"⚠️ Stream error: {e}")

        stream_thread = threading.Thread(target=run_stream, daemon=True)
        stream_thread.start()

        await websocket.send_json(
            {"type": "ready", "message": "Ready to receive audio"}
        )

        # Process incoming audio chunks
        while is_connected:
            try:
                # Wait for message with timeout
                data = await asyncio.wait_for(websocket.receive(), timeout=30.0)

                if data.get("type") == "websocket.disconnect":
                    break

                if "bytes" in data:
                    # Audio chunk received - send to AssemblyAI via queue
                    audio_data = data["bytes"]
                    audio_stream.write(audio_data)
                elif "text" in data:
                    # Handle text commands
                    msg = json.loads(data["text"])
                    if msg.get("type") == "stop":
                        print("⏹️ Stop command received")
                        break

                # Check if we have a final transcript from turn detection
                if transcript_ready.is_set():
                    break

            except asyncio.TimeoutError:
                print("⏰ WebSocket timeout - no audio received")
                break
            except WebSocketDisconnect:
                print("🔌 WebSocket disconnected by client")
                break

        # Stop the audio stream
        if audio_stream:
            audio_stream.stop()

        # Wait for stream thread to finish
        if stream_thread and stream_thread.is_alive():
            stream_thread.join(timeout=2.0)

        # Disconnect from AssemblyAI
        if client:
            try:
                client.disconnect(terminate=True)
            except Exception as e:
                print(f"⚠️ Error disconnecting client: {e}")

        # Send final result
        if final_transcript and is_connected:
            await websocket.send_json({"type": "final", "text": final_transcript})
            print(f"📤 Sent final transcript: {final_transcript}")

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        import traceback

        traceback.print_exc()
        if is_connected:
            try:
                await websocket.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
    finally:
        is_connected = False
        if audio_stream:
            audio_stream.stop()
        if client:
            try:
                client.disconnect(terminate=True)
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
        print("🔒 WebSocket connection closed")
