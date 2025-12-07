# Voice Chat Web App with X AI

A simple web application that allows you to have voice conversations with Grok LLM.

## Features

- 🎤 Voice input recording
- 🤖 Grok LLM integration
- 🔊 Text-to-speech audio responses
- 💬 Real-time conversation interface

## Setup

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or use the helper script
./activate.sh
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Variables

Create a `.env` file in the project root with:

```bash
XAI_API_KEY=your-xai-api-key
ASSEMBLYAI_API_KEY=your-assemblyai-api-key
# OR use ASSEMBLY_API_KEY (both are supported)
ASSEMBLY_API_KEY=your-assemblyai-api-key
X_BEARER_TOKEN=your-x-bearer-token
```

**Getting API Keys:**
- **XAI_API_KEY**: Get from [x.ai](https://x.ai)
- **ASSEMBLYAI_API_KEY** or **ASSEMBLY_API_KEY**: Get from [AssemblyAI](https://www.assemblyai.com/) (free tier available)
- **X_BEARER_TOKEN**: Get from [Twitter Developer Portal](https://developer.twitter.com/) - you need a Bearer Token for API v2

### 4. Run the Application

Make sure your virtual environment is activated, then:

```bash
python app.py
```

**Note:** Always activate the virtual environment before running the app:
```bash
source venv/bin/activate
python app.py
```

Then open your browser to: `http://localhost:5000`

## How It Works

### Podcast Interruption Feature

1. **Upload Podcast**: 
   - **Option 1**: Paste a URL (YouTube, podcast RSS feed, or direct audio link)
   - **Option 2**: Upload a podcast audio file
   - Specify the speaker name (e.g., "Joe Rogan")
2. **Transcribe**: The podcast is downloaded (if URL) and transcribed using AssemblyAI with speaker diarization
3. **Ask Questions**: Record your voice to ask questions about the podcast
4. **X Search**: The system searches X (Twitter) for tweets from the speaker related to your question
5. **Context-Aware Response**: Grok LLM responds using:
   - The podcast transcript
   - Recent tweets from the speaker
   - Your question
6. **Voice Cloned Response**: The response is generated using Grok Voice with the speaker's voice (if available)

### Example Flow

1. Upload a Joe Rogan podcast
2. Joe says: "I think AI is going to change everything..."
3. You interrupt: "What do you think about AI safety?"
4. System searches Joe Rogan's tweets about AI
5. Grok responds in Joe's voice style using podcast context + his tweets

## Notes

- Make sure to allow microphone access when prompted
- The app uses Google Speech Recognition for transcription
- Voice files from the demo directory are automatically used if available
- **URL Support**: The app uses `yt-dlp` to download audio from:
  - YouTube videos
  - Podcast RSS feeds
  - Direct audio file URLs (MP3, M4A, etc.)
- If `yt-dlp` is not installed, the app will attempt direct download for audio URLs

# GrokRere
