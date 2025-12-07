#!/usr/bin/env python3
"""
Flask web app for Voice Chat with Grok LLM and Podcast Interruption
"""

import os
import base64
import requests
import tempfile
import subprocess
import time
from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
from werkzeug.utils import secure_filename
from urllib.parse import urlparse

# AssemblyAI Python SDK
try:
    import assemblyai as aai
except ImportError:
    aai = None

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# API Configuration
XAI_API_KEY = os.environ.get("XAI_API_KEY")
# Support both ASSEMBLYAI_API_KEY and ASSEMBLY_API_KEY for flexibility
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY") or os.environ.get("ASSEMBLY_API_KEY")
X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN")  # For X API v2 search
TTS_BASE_URL = "https://us-east-4.api.x.ai/voice-staging"
TTS_ENDPOINT = f"{TTS_BASE_URL}/api/v1/text-to-speech/generate"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Initialize AssemblyAI SDK
if aai and ASSEMBLYAI_API_KEY:
    aai.settings.api_key = ASSEMBLYAI_API_KEY

MAX_INPUT_LENGTH = 4096

# Store podcast data in memory (in production, use a database)
podcast_data = {}


def file_to_base64(file_path: str) -> str:
    """Convert a file to base64 string."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def tts_request(input_text: str, voice_file: str | None = None) -> bytes | None:
    """Generate speech from text using the TTS API (from demo.py)."""
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


def download_audio_from_url(url: str) -> str | None:
    """Download audio from URL using yt-dlp. Returns path to downloaded file."""
    try:
        # Create temp directory for downloaded audio
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, '%(title)s.%(ext)s')
        
        # Use yt-dlp to download audio
        # -x: extract audio only
        # --audio-format mp3: convert to mp3
        # -o: output file template
        cmd = [
            'yt-dlp',
            '-x',  # Extract audio only
            '--audio-format', 'mp3',
            '--audio-quality', '0',  # Best quality
            '-o', output_template,
            '--no-playlist',  # Don't download playlists
            url
        ]
        
        print(f"Downloading audio from URL: {url}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"yt-dlp error: {result.stderr}")
            print(f"yt-dlp stdout: {result.stdout}")
            # Try direct download if yt-dlp fails (for direct audio URLs)
            try:
                print("Attempting direct download...")
                response = requests.get(url, stream=True, timeout=60, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code == 200:
                    output_path = os.path.join(temp_dir, 'audio.mp3')
                    with open(output_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Direct download successful: {output_path}")
                    return output_path
            except Exception as e:
                print(f"Direct download also failed: {e}")
            return None
        
        # Find the downloaded file (yt-dlp may add extension)
        downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith(('.mp3', '.m4a', '.webm', '.ogg'))]
        
        if downloaded_files:
            output_path = os.path.join(temp_dir, downloaded_files[0])
            # If it's not mp3, we might need to convert it, but for now return as-is
            # AssemblyAI can handle various formats
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Download successful: {output_path} ({os.path.getsize(output_path)} bytes)")
                return output_path
            else:
                print("Downloaded file is empty")
                return None
        else:
            print("No audio file found after download")
            print(f"Files in temp dir: {os.listdir(temp_dir)}")
            return None
            
    except subprocess.TimeoutExpired:
        print("Download timeout")
        return None
    except FileNotFoundError:
        print("yt-dlp not found. Install with: pip install yt-dlp")
        # Fallback to direct download
        try:
            print("Attempting direct download (yt-dlp not available)...")
            temp_dir = tempfile.mkdtemp()
            response = requests.get(url, stream=True, timeout=60, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            if response.status_code == 200:
                output_path = os.path.join(temp_dir, 'audio.mp3')
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Direct download successful: {output_path}")
                return output_path
        except Exception as e:
            print(f"Direct download failed: {e}")
        return None
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_audio_to_wav(input_path: str, output_path: str = None) -> str | None:
    """Convert audio file to WAV format for speech recognition."""
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.wav'
    
    try:
        # Try using pydub first (simpler)
        if PYDUB_AVAILABLE:
            audio = AudioSegment.from_file(input_path)
            audio.export(output_path, format="wav")
            return output_path
        
        # Fallback to ffmpeg
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ar', '16000',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        else:
            print(f"ffmpeg conversion failed: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("ffmpeg not found. Install with: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
        # If no converter available, try to use file as-is
        return input_path
    except Exception as e:
        print(f"Audio conversion error: {e}")
        # Try to use original file
        return input_path


def extract_voice_sample_from_podcast(podcast_path: str, duration_seconds: int = 30) -> str | None:
    """Extract a voice sample from the podcast for voice cloning."""
    try:
        # Create temp file for voice sample - use mp3 format (more universally supported)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            sample_path = tmp_file.name
        
        # Extract first N seconds using ffmpeg
        # Use mp3 format with libmp3lame codec for better compatibility
        cmd = [
            'ffmpeg',
            '-i', podcast_path,
            '-t', str(duration_seconds),  # Duration in seconds
            '-acodec', 'libmp3lame',  # Use mp3 codec
            '-ab', '192k',  # Audio bitrate
            '-ar', '44100',  # Sample rate
            '-y',  # Overwrite
            sample_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(sample_path):
            file_size = os.path.getsize(sample_path)
            if file_size > 0:
                print(f"Extracted voice sample: {sample_path} ({file_size} bytes)")
                return sample_path
            else:
                print(f"Extracted file is empty, trying alternative method...")
        else:
            print(f"ffmpeg extraction failed: {result.stderr[:200]}")
        
        # If ffmpeg fails, try pydub
        if PYDUB_AVAILABLE:
            try:
                print("Trying pydub extraction...")
                audio = AudioSegment.from_file(podcast_path)
                sample = audio[:duration_seconds * 1000]  # Convert to milliseconds
                # Use mp3 format instead of m4a for better compatibility
                sample.export(sample_path, format="mp3", bitrate="192k")
                if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
                    print(f"Pydub extraction successful: {sample_path}")
                    return sample_path
            except Exception as e:
                print(f"Pydub extraction failed: {e}")
        
        # If all else fails, return the original podcast file
        print("Could not extract sample, using full podcast file")
        # Clean up failed sample file
        if os.path.exists(sample_path):
            try:
                os.unlink(sample_path)
            except:
                pass
        return podcast_path
        
    except FileNotFoundError:
        print("ffmpeg not found, using full podcast file for voice cloning")
        return podcast_path
    except Exception as e:
        print(f"Error extracting voice sample: {e}")
        import traceback
        traceback.print_exc()
        return podcast_path


def transcribe_podcast_with_assemblyai(audio_file_path: str) -> dict | None:
    """Transcribe podcast audio using AssemblyAI Python SDK."""
    if not ASSEMBLYAI_API_KEY:
        print("ASSEMBLYAI_API_KEY not set")
        return None
    
    if not aai:
        print("AssemblyAI SDK not installed. Install with: pip install assemblyai")
        return None

    try:
        # Create transcriber
        transcriber = aai.Transcriber()
        
        # Configure transcription with speaker diarization
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            language_code="en"  # Adjust if needed
        )
        
        # Transcribe the audio file
        print(f"Starting transcription of {audio_file_path}...")
        transcript = transcriber.transcribe(audio_file_path, config=config)
        
        # Check if transcription completed successfully
        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription error: {transcript.error}")
            return None
        
        # Convert transcript to dictionary format for compatibility
        transcript_dict = {
            'id': transcript.id,
            'status': transcript.status.value if hasattr(transcript.status, 'value') else str(transcript.status),
            'text': transcript.text,
            'utterances': []
        }
        
        # Extract utterances with speaker labels
        if transcript.utterances:
            for utterance in transcript.utterances:
                transcript_dict['utterances'].append({
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    'start': utterance.start,
                    'end': utterance.end
                })
        
        print(f"Transcription completed successfully")
        return transcript_dict
            
    except Exception as e:
        print(f"Error transcribing with AssemblyAI: {e}")
        import traceback
        traceback.print_exc()
        return None


def search_x_tweets(username: str, query: str = "", max_results: int = 10) -> list:
    """Search for tweets from a specific user on X (Twitter)."""
    if not X_BEARER_TOKEN:
        return []
    
    try:
        # First, get user ID from username
        user_lookup_url = f"https://api.twitter.com/2/users/by/username/{username}"
        user_response = requests.get(
            user_lookup_url,
            headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        )
        
        if user_response.status_code != 200:
            print(f"User lookup failed: {user_response.text}")
            return []
        
        user_id = user_response.json()['data']['id']
        
        # Search for tweets
        search_params = {
            "query": f"from:{username} {query}" if query else f"from:{username}",
            "max_results": max_results,
            "tweet.fields": "created_at,text,author_id"
        }
        
        search_response = requests.get(
            X_SEARCH_URL,
            params=search_params,
            headers={"Authorization": f"Bearer {X_BEARER_TOKEN}"}
        )
        
        if search_response.status_code == 200:
            data = search_response.json()
            return data.get('data', [])
        else:
            print(f"Search failed: {search_response.text}")
            return []
            
    except Exception as e:
        print(f"Error searching X: {e}")
        return []


def call_grok_llm(user_input: str, podcast_transcript: str = "", speaker_name: str = "", x_tweets: list = None) -> str:
    """Send user input to Grok LLM with podcast context and X search results."""
    if not XAI_API_KEY:
        print("❌ XAI_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build context-aware system prompt
    system_parts = []
    
    if speaker_name:
        system_parts.append(f"You are {speaker_name}. Respond to the user's question in {speaker_name}'s voice and style.")
    else:
        system_parts.append("You are a helpful AI assistant.")
    
    if podcast_transcript:
        # Limit transcript to avoid token limits, but keep it informative
        transcript_snippet = podcast_transcript[:3000] if len(podcast_transcript) > 3000 else podcast_transcript
        system_parts.append(f"\n\nCONTEXT FROM PODCAST TRANSCRIPT:\n{transcript_snippet}")
        system_parts.append("\n\nIMPORTANT: Use the podcast transcript as CONTEXT to answer the user's question. Do NOT just repeat the transcript. Answer the user's question based on what was discussed in the podcast.")
    
    if x_tweets:
        tweets_text = "\n".join([f"- {tweet.get('text', '')}" for tweet in x_tweets[:5]])
        system_parts.append(f"\n\nRECENT TWEETS FROM {speaker_name.upper()}:\n{tweets_text}")
        system_parts.append(f"\n\nUse these tweets to understand {speaker_name}'s speaking style and opinions.")
    
    system_parts.append("\n\nCRITICAL: Answer the user's question directly. Do not just quote the transcript. Provide a thoughtful response in the style of the speaker.")
    
    system_prompt = "\n".join(system_parts)
    
    print(f"📝 System prompt length: {len(system_prompt)} chars")
    print(f"📝 User input: {user_input[:100]}...")
    
    data = {
        "model": "grok-beta",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 1500  # Increased for longer responses
    }

    try:
        print(f"🌐 Calling Grok API at {GROK_API_URL}...")
        response = requests.post(GROK_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            grok_text = result["choices"][0]["message"]["content"]
            print(f"✅ Grok API response received: {len(grok_text)} characters")
            print(f"📄 Response preview: {grok_text[:200]}...")
            return grok_text
        else:
            print(f"❌ Grok API error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        print("❌ Grok API timeout")
        return None
    except Exception as e:
        print(f"❌ Error calling Grok API: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/')
def index():
    """Main page with voice chat interface."""
    return render_template('index.html')


@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """Transcribe user voice input to text."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Save to temporary file (keep original extension)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp_file:
            audio_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
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
                audio = recognizer.record(source)
            
            text = recognizer.recognize_google(audio)
            return jsonify({"text": text})
        finally:
            # Clean up temp files
            for path in [tmp_path, wav_path if 'wav_path' in locals() and wav_path != tmp_path else None]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
                
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio"}), 400
    except Exception as e:
        print(f"Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload-podcast', methods=['POST'])
def upload_podcast():
    """Upload and transcribe podcast audio from file or URL."""
    speaker_name = request.form.get('speaker_name', '').strip()
    if not speaker_name:
        return jsonify({"error": "Speaker name is required"}), 400
    
    podcast_url = request.form.get('podcast_url', '').strip()
    podcast_file = request.files.get('podcast')
    
    filepath = None
    cleanup_file = False
    
    try:
        # Handle URL or file upload
        if podcast_url:
            # Download from URL
            print(f"Downloading podcast from URL: {podcast_url}")
            filepath = download_audio_from_url(podcast_url)
            if not filepath:
                return jsonify({"error": "Failed to download audio from URL. Check server logs for details."}), 500
            cleanup_file = True
            # Use URL hash for unique ID
            podcast_id = f"url_{abs(hash(podcast_url)) % 100000}"
        elif podcast_file:
            # Handle file upload
            filename = secure_filename(podcast_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            podcast_file.save(filepath)
            podcast_id = filename
        else:
            return jsonify({"error": "Either podcast file or URL is required"}), 400
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({"error": f"Failed to get audio file. Filepath: {filepath}"}), 500
        
        # Verify file is readable and has content
        file_size = os.path.getsize(filepath)
        print(f"Audio file ready: {filepath} ({file_size} bytes)")
        
        if file_size == 0:
            return jsonify({"error": "Downloaded file is empty"}), 500
        
        # Transcribe with AssemblyAI
        print(f"Starting transcription of {filepath}...")
        transcript_data = transcribe_podcast_with_assemblyai(filepath)
        
        if not transcript_data:
            return jsonify({"error": "Failed to transcribe podcast"}), 500
        
        # Extract transcript text and speaker information
        transcript_text = transcript_data.get('text', '')
        utterances = transcript_data.get('utterances', [])
        
        # Format transcript with speaker labels
        formatted_transcript = ""
        for utterance in utterances:
            speaker = utterance.get('speaker', 'Unknown')
            text = utterance.get('text', '')
            formatted_transcript += f"Speaker {speaker}: {text}\n"
        
        # Extract voice sample from podcast for voice cloning
        voice_sample_path = None
        if filepath and os.path.exists(filepath):
            print("Extracting voice sample from podcast...")
            voice_sample_path = extract_voice_sample_from_podcast(filepath, duration_seconds=30)
        
        # Store podcast data
        podcast_data[podcast_id] = {
            'speaker_name': speaker_name,
            'transcript': transcript_text,
            'formatted_transcript': formatted_transcript,
            'utterances': utterances,
            'podcast_file_path': filepath,  # Store original file path
            'voice_sample_path': voice_sample_path  # Store voice sample for cloning
        }
        
        # IMPORTANT: Keep the podcast file for voice cloning!
        # Don't delete it - we need it to generate responses in the podcast host's voice
        if cleanup_file:
            # For URL downloads, move to uploads folder so it persists for voice cloning
            try:
                uploads_dir = app.config['UPLOAD_FOLDER']
                os.makedirs(uploads_dir, exist_ok=True)
                filename = os.path.basename(filepath) or f"podcast_{podcast_id}.mp3"
                persistent_path = os.path.join(uploads_dir, filename)
                
                # Only move if not already in uploads folder
                if not filepath.startswith(uploads_dir):
                    import shutil
                    if os.path.exists(filepath):
                        shutil.move(filepath, persistent_path)
                        filepath = persistent_path
                        # Update stored path
                        podcast_data[podcast_id]['podcast_file_path'] = filepath
                        print(f"✅ Moved podcast file to persistent location: {filepath}")
            except Exception as e:
                print(f"Error moving podcast file (keeping original): {e}")
                # Keep original file if move fails
                pass
        # Uploaded files are already in the right place, keep them
        
        return jsonify({
            "podcast_id": podcast_id,
            "speaker_name": speaker_name,
            "transcript_preview": transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text,
            "message": "Podcast transcribed successfully"
        })
        
    except Exception as e:
        print(f"Upload error: {e}")
        # Clean up on error
        if cleanup_file and filepath and os.path.exists(filepath):
            try:
                os.unlink(filepath)
            except:
                pass
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat request: send to Grok with podcast context and return audio response."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        user_input = data.get('text', '')
        podcast_id = data.get('podcast_id', '')
        
        print(f"📨 Chat request - User input: {user_input[:50]}..., Podcast ID: {podcast_id}")
        
        if not user_input:
            return jsonify({"error": "No text provided"}), 400
        
        # Get podcast context
        podcast_transcript = ""
        speaker_name = ""
        if podcast_id:
            if podcast_id in podcast_data:
                podcast_info = podcast_data[podcast_id]
                podcast_transcript = podcast_info.get('formatted_transcript', '')
                speaker_name = podcast_info.get('speaker_name', '')
                print(f"📻 Found podcast context - Speaker: {speaker_name}, Transcript length: {len(podcast_transcript)}")
            else:
                print(f"⚠️  Podcast ID {podcast_id} not found in podcast_data. Available IDs: {list(podcast_data.keys())}")
        
        # Search X for speaker tweets
        x_tweets = []
        if speaker_name:
            # Try different username variations
            base_username = speaker_name.lower().replace(' ', '').replace('-', '')
            username_variations = [
                base_username,
                f"{base_username}official",
                f"{base_username}podcast",
                f"{base_username}show"
            ]
            
            # Try each variation until we find tweets
            for username in username_variations:
                x_tweets = search_x_tweets(username, query=user_input[:50], max_results=5)
                if x_tweets:
                    print(f"🐦 Found {len(x_tweets)} tweets from @{username}")
                    break
        
        # Call Grok LLM with context
        print("🤖 Calling Grok LLM...")
        grok_response = call_grok_llm(
            user_input, 
            podcast_transcript=podcast_transcript,
            speaker_name=speaker_name,
            x_tweets=x_tweets
        )
        
        if not grok_response:
            return jsonify({"error": "Failed to get response from Grok"}), 500
        
        print(f"✅ Grok response received: {grok_response[:100]}...")
        
        # Use the podcast audio file as the voice file (like demo.py pattern)
        voice_file = None
        
        # If podcast_id provided but not found, try to use Joe Rogan voice as fallback
        if podcast_id and podcast_id not in podcast_data:
            print(f"⚠️  Podcast ID {podcast_id} not found. Using Joe Rogan voice as fallback.")
            # Try to get speaker name from the request or use "Joe Rogan" as default
            if not speaker_name:
                speaker_name = "Joe Rogan"
        
        if podcast_id and podcast_id in podcast_data:
            podcast_info = podcast_data[podcast_id]
            
            # Priority 1: Use the original podcast file (best quality, full voice)
            podcast_file_path = podcast_info.get('podcast_file_path')
            if podcast_file_path and os.path.exists(podcast_file_path):
                voice_file = podcast_file_path
                print(f"✅ Using podcast audio file for voice cloning: {voice_file}")
            # Priority 2: Use extracted voice sample if original not available
            else:
                voice_sample = podcast_info.get('voice_sample_path')
                if voice_sample and os.path.exists(voice_sample):
                    voice_file = voice_sample
                    print(f"✅ Using podcast voice sample: {voice_sample}")
        
        # Fallback: Try voices directory (especially joe-rogan.mp3)
        if not voice_file:
            voices_dir = os.path.join(os.path.dirname(__file__), '..', 'voice-demo-hackathon 2', 'voices')
            if os.path.exists(voices_dir):
                # Try to find a voice file matching the speaker name
                if speaker_name:
                    speaker_lower = speaker_name.lower().replace(' ', '-')
                    # Try multiple variations: "joe rogan" -> "joe-rogan", "joerogan", etc.
                    name_variations = [
                        speaker_lower,
                        speaker_name.lower().replace(' ', ''),
                        speaker_name.lower().replace(' ', '_')
                    ]
                    
                    for name_var in name_variations:
                        possible_files = [
                            f"{name_var}.m4a",
                            f"{name_var}.mp3",
                            f"{name_var}.wav"
                        ]
                        for filename in possible_files:
                            filepath = os.path.join(voices_dir, filename)
                            if os.path.exists(filepath):
                                voice_file = filepath
                                print(f"✅ Using voice file from voices directory: {voice_file}")
                                break
                        if voice_file:
                            break
                
                # Fallback: Try joe-rogan.mp3 specifically (common case)
                if not voice_file:
                    joe_rogan_file = os.path.join(voices_dir, 'joe-rogan.mp3')
                    if os.path.exists(joe_rogan_file):
                        voice_file = joe_rogan_file
                        print(f"✅ Using Joe Rogan voice file: {voice_file}")
                
                # Final fallback to any available voice file
                if not voice_file:
                    voice_files = [f for f in os.listdir(voices_dir) if f.endswith(('.m4a', '.mp3', '.wav'))]
                    if voice_files:
                        voice_file = os.path.join(voices_dir, voice_files[0])
                        print(f"Using fallback voice file: {voice_file}")
        
        if not voice_file:
            print("⚠️  Warning: No voice file found, TTS will use default voice")
        else:
            print(f"🎤 Generating response with voice from: {os.path.basename(voice_file)}")
        
        print("🎵 Generating TTS audio...")
        audio_data = tts_request(grok_response, voice_file)
        
        if not audio_data:
            return jsonify({"error": "Failed to generate audio"}), 500
        
        print(f"✅ Audio generated: {len(audio_data)} bytes")
        
        # Return audio as base64
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return jsonify({
            "text": grok_response,
            "audio": audio_base64
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
