from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(video_url: str) -> str:
    url = video_url.strip()
    parsed = urlparse(url)

    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.lstrip("/")
        if not video_id:
            raise ValueError("Invalid youtu.be URL")
        return video_id

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            if not video_id:
                raise ValueError("Missing video id (v=)")
            return video_id

        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/shorts/")[1].split("/")[0]
            if not video_id:
                raise ValueError("Invalid Shorts URL")
            return video_id

    raise ValueError("Invalid YouTube URL")


def fetch_transcript_text(video_id: str, preferred_languages=None) -> Tuple[str, Optional[list]]:
    preferred_languages = preferred_languages or ["en", "en-US", "en-GB"]
    ytt_api = YouTubeTranscriptApi()

    try:
        fetched = ytt_api.fetch(video_id, languages=preferred_languages)
        snippets = list(fetched)
        text = " ".join(s.text for s in snippets)
        return text, snippets
    except Exception:
        pass

    transcript_list = ytt_api.list(video_id)
    first_transcript = next(iter(transcript_list))
    fetched = first_transcript.fetch()
    snippets = list(fetched)
    text = " ".join(s.text for s in snippets)
    return text, snippets


def safe_fetch_transcript(video_url: str) -> Tuple[str, str]:
    try:
        video_id = extract_video_id(video_url)
        transcript_text, _ = fetch_transcript_text(video_id)

        if not transcript_text.strip():
            return video_id, "No transcript available for this video."

        return video_id, transcript_text

    except Exception as e:
        st.error(f"Transcript error: {type(e).__name__}: {e}")
        return "Unknown", "No transcript available for this video."
