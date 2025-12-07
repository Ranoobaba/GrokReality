"""
Transcription service using AssemblyAI.
"""
from typing import Optional
from config import ASSEMBLYAI_API_KEY

# AssemblyAI Python SDK
try:
    import assemblyai as aai
    if ASSEMBLYAI_API_KEY:
        aai.settings.api_key = ASSEMBLYAI_API_KEY
except ImportError:
    aai = None


def transcribe_podcast_with_assemblyai(audio_file_path: str) -> Optional[dict]:
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

