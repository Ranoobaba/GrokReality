"""
Grok LLM service for handling API calls.
"""

from typing import Optional

import requests

from config import GROK_API_URL, GROK_MODEL, XAI_API_KEY


def call_grok_llm(
    user_input: str,
    podcast_transcript: str = "",
    speaker_name: str = "",
    x_tweets: Optional[list] = None,
) -> Optional[str]:
    """Send user input to Grok LLM with podcast context and X search results."""
    if not XAI_API_KEY:
        print("❌ XAI_API_KEY not set")
        return None

    headers = {
        "Authorization": f"Bearer {XAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build context-aware system prompt - CONCISE AND CONVERSATIONAL
    system_parts = []

    if speaker_name:
        system_parts.append(
            f"You are {speaker_name}. Respond briefly in {speaker_name}'s voice and style."
        )
    else:
        system_parts.append("You are a helpful AI assistant.")

    if podcast_transcript:
        # Limit transcript to avoid token limits, but keep it informative
        transcript_snippet = (
            podcast_transcript[:2000]
            if len(podcast_transcript) > 2000
            else podcast_transcript
        )
        system_parts.append(f"\n\nPODCAST CONTEXT:\n{transcript_snippet}")

    if x_tweets:
        tweets_text = "\n".join(
            [f"- {tweet.get('text', '')}" for tweet in x_tweets[:3]]
        )
        system_parts.append(f"\n\nRECENT TWEETS:\n{tweets_text}")

    # CRITICAL: Keep responses SHORT and CONVERSATIONAL
    system_parts.append("""

CRITICAL INSTRUCTIONS:
- Keep your response to 1-3 sentences MAX. This is a CONVERSATION, not a lecture.
- Be direct and casual, like you're chatting with a friend.
- Don't repeat the question back. Don't use filler phrases like "Great question!" or "That's interesting."
- If you don't know something, just say so briefly.
- Match the energy and tone of casual podcast banter.""")

    system_prompt = "\n".join(system_parts)

    print(f"📝 System prompt length: {len(system_prompt)} chars")
    print(f"📝 User input: {user_input[:100]}...")

    data = {
        "model": GROK_MODEL,  # Using updated model from config
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        "temperature": 0.8,  # Slightly higher for more natural conversation
        "max_tokens": 150,  # Short, conversational responses
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
