"""
Configuration settings for the FastAPI application.
"""

import os

from dotenv import load_dotenv

# Load environment variables from .env file
try:
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use system env vars

# App Configuration
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size
UPLOAD_FOLDER = "uploads"

# API Configuration
XAI_API_KEY = os.environ.get("XAI_API_KEY")
# Support both ASSEMBLYAI_API_KEY and ASSEMBLY_API_KEY for flexibility
ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY") or os.environ.get(
    "ASSEMBLY_API_KEY"
)
X_BEARER_TOKEN = os.environ.get("X_BEARER_TOKEN")  # For X API v2 search

# API Endpoints
TTS_BASE_URL = "https://us-east-4.api.x.ai/voice-staging"
TTS_ENDPOINT = f"{TTS_BASE_URL}/api/v1/text-to-speech/generate"
GROK_API_URL = "https://api.x.ai/v1/chat/completions"
X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# Grok Model Configuration
GROK_MODEL = "grok-2-latest"  # Updated from deprecated grok-beta

# Other Configuration
MAX_INPUT_LENGTH = 4096
