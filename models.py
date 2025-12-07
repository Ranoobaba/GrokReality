"""
Pydantic models for request/response validation.
"""

from typing import Optional

from pydantic import BaseModel, Field

# ============ Request Models ============


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    text: str = Field(..., min_length=1, description="User's message text")
    podcast_id: Optional[str] = Field(
        default="", description="ID of the uploaded podcast for context"
    )


# ============ Response Models ============


class TranscribeResponse(BaseModel):
    """Response model for transcribe endpoint."""

    text: str = Field(..., description="Transcribed text from audio")


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error message")


class UploadPodcastResponse(BaseModel):
    """Response model for podcast upload endpoint."""

    podcast_id: str = Field(
        ..., description="Unique identifier for the uploaded podcast"
    )
    speaker_name: str = Field(..., description="Name of the speaker")
    transcript_preview: str = Field(..., description="Preview of the transcript")
    message: str = Field(..., description="Success message")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    text: str = Field(..., description="Grok's response text")
    audio: str = Field(..., description="Base64-encoded audio response")


# ============ Internal Models ============


class Utterance(BaseModel):
    """Model for a single utterance in a transcript."""

    speaker: str
    text: str
    start: Optional[int] = None
    end: Optional[int] = None


class PodcastData(BaseModel):
    """Model for stored podcast data."""

    speaker_name: str
    transcript: str
    formatted_transcript: str
    utterances: list[Utterance] = []
    podcast_file_path: Optional[str] = None
    voice_sample_path: Optional[str] = None
