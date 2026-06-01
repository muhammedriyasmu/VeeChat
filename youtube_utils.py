import os
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import (
    AgeRestricted,
    CouldNotRetrieveTranscript,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
    VideoUnavailable,
    VideoUnplayable,
)
from youtube_transcript_api.proxies import GenericProxyConfig


@dataclass(frozen=True)
class TranscriptResult:
    video_id: str
    text: str
    status: str
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"


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


def _make_transcript_api() -> YouTubeTranscriptApi:
    proxy_url = os.getenv("YOUTUBE_PROXY_URL", "").strip()
    http_proxy = os.getenv("YOUTUBE_HTTP_PROXY", "").strip()
    https_proxy = os.getenv("YOUTUBE_HTTPS_PROXY", "").strip()

    if proxy_url:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(http_url=proxy_url, https_url=proxy_url)
        )

    if http_proxy or https_proxy:
        return YouTubeTranscriptApi(
            proxy_config=GenericProxyConfig(
                http_url=http_proxy or None,
                https_url=https_proxy or None,
            )
        )

    return YouTubeTranscriptApi()


def fetch_transcript_text(video_id: str, preferred_languages=None) -> Tuple[str, Optional[list]]:
    preferred_languages = preferred_languages or ["en", "en-US", "en-GB"]
    ytt_api = _make_transcript_api()

    try:
        fetched = ytt_api.fetch(video_id, languages=preferred_languages)
        snippets = list(fetched)
        text = " ".join(s.text for s in snippets)
        return text, snippets
    except (IpBlocked, RequestBlocked):
        raise
    except (NoTranscriptFound, TranscriptsDisabled):
        pass

    transcript_list = ytt_api.list(video_id)
    first_transcript = next(iter(transcript_list))
    fetched = first_transcript.fetch()
    snippets = list(fetched)
    text = " ".join(s.text for s in snippets)
    return text, snippets


def safe_fetch_transcript(video_url: str) -> TranscriptResult:
    try:
        video_id = extract_video_id(video_url)
        transcript_text, _ = fetch_transcript_text(video_id)

        if not transcript_text.strip():
            return TranscriptResult(
                video_id=video_id,
                text="",
                status="missing",
                message="No transcript text was returned for this video.",
            )

        return TranscriptResult(video_id=video_id, text=transcript_text, status="ok")

    except (IpBlocked, RequestBlocked) as e:
        return TranscriptResult(
            video_id=getattr(e, "video_id", "Unknown"),
            text="",
            status="blocked",
            message=(
                "YouTube blocked transcript requests from this network. "
                "Paste a transcript manually below, run the app from another network, "
                "or configure YOUTUBE_PROXY_URL / YOUTUBE_HTTP_PROXY / YOUTUBE_HTTPS_PROXY."
            ),
        )
    except (NoTranscriptFound, TranscriptsDisabled):
        return TranscriptResult(
            video_id=video_id,
            text="",
            status="missing",
            message="This video does not expose a transcript through YouTube captions.",
        )
    except AgeRestricted:
        return TranscriptResult(
            video_id=video_id,
            text="",
            status="error",
            message="This video is age restricted, so captions cannot be retrieved automatically.",
        )
    except (VideoUnavailable, VideoUnplayable):
        return TranscriptResult(
            video_id=video_id,
            text="",
            status="error",
            message="This video is unavailable or unplayable.",
        )
    except CouldNotRetrieveTranscript as e:
        return TranscriptResult(
            video_id=getattr(e, "video_id", "Unknown"),
            text="",
            status="error",
            message=f"Transcript error: {type(e).__name__}.",
        )
    except Exception as e:
        return TranscriptResult(
            video_id="Unknown",
            text="",
            status="error",
            message=f"Transcript error: {type(e).__name__}: {e}",
        )
