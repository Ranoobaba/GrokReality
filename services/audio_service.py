"""
Audio processing utilities.
"""
import os
import tempfile
import subprocess
import requests
from typing import Optional

# Audio processing
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


def convert_audio_to_wav(input_path: str, output_path: Optional[str] = None) -> Optional[str]:
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


def extract_voice_sample_from_podcast(podcast_path: str, duration_seconds: int = 30) -> Optional[str]:
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


def download_audio_from_url(url: str) -> Optional[str]:
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

