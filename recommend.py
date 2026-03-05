# recommend.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
import requests


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "you", "we", "they", "i", "he", "she", "them", "our", "your", "my", "at", "by",
    "from", "not", "do", "does", "did", "can", "could", "should", "would", "will", "just",
    "about", "into", "over", "than", "also", "when", "what", "why", "how", "who", "which",
}

# Words must be >= 4 chars to avoid noise
WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _top_keywords(text: str, k: int = 12) -> List[str]:
    """
    Simple keyword extraction:
    - keep only alphabetic words >= 4 chars
    - remove stopwords
    - score by frequency
    """
    text = (text or "").lower()
    words = WORD_RE.findall(text)

    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1

    # Sort by frequency then alphabetically for stability
    top = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:k]
    return [w for w, _ in top]


def _build_query(
    transcript_text: str,
    seed_title: Optional[str] = None,
    seed_channel: Optional[str] = None,
    max_terms: int = 7,
) -> str:
    """
    Build a decent YouTube search query based on:
    - top transcript keywords
    - optional title/channel hint (helps relevance)
    """
    kws = _top_keywords(transcript_text, k=12)

    parts: List[str] = []
    if seed_title:
        parts.append(_normalize_space(seed_title))
    if seed_channel:
        parts.append(_normalize_space(seed_channel))

    # Add top keywords
    parts.extend(kws[:max_terms])

    q = _normalize_space(" ".join(parts))
    return q if q else "tutorial"


def _dedupe_by_url(items: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for it in items:
        url = (it.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(it)
    return out

def fallback_search_links(
    transcript_text: str,
    video_url: str,
    seed_title: Optional[str] = None,
) -> List[Dict]:
    """
    No API key? Return useful YouTube search links + current video link.
    """
    q_strict = _build_query(transcript_text, seed_title=seed_title, max_terms=6)
    q_broad = " ".join(_top_keywords(transcript_text, k=3)) or "learning"

    q1 = requests.utils.quote(q_strict)
    q2 = requests.utils.quote(q_broad)

    links = [
        {
            "title": "Search similar videos (based on transcript keywords)",
            "channel": "YouTube Search",
            "url": f"https://www.youtube.com/results?search_query={q1}",
            "thumbnail": None,
        },
        {
            "title": "More videos on this topic (broader search)",
            "channel": "YouTube Search",
            "url": f"https://www.youtube.com/results?search_query={q2}",
            "thumbnail": None,
        },
        {
            "title": "Open the current video",
            "channel": "Current",
            "url": video_url,
            "thumbnail": None,
        },
    ]
    return _dedupe_by_url(links)


def related_videos_youtube_api(
    youtube_api_key: str,
    transcript_text: str,
    max_results: int = 8,
    region_code: str = "IN",
    relevance_language: str = "en",
    seed_title: Optional[str] = None,
    seed_channel: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch related videos using YouTube Data API (search endpoint).
    Returns list of dicts: {title, channel, url, thumbnail}

    NOTE:
    - This uses transcript keywords to form a query.
    - It's NOT "relatedToVideoId" (which needs a videoId). This is topic-based.
    """
    if not youtube_api_key or not youtube_api_key.strip():
        return []

    q = _build_query(
        transcript_text,
        seed_title=seed_title,
        seed_channel=seed_channel,
        max_terms=7,
    )

    endpoint = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "type": "video",
        "maxResults": max(1, min(int(max_results), 25)),
        "q": q,
        "key": youtube_api_key.strip(),
        "regionCode": region_code,
        "relevanceLanguage": relevance_language,
        "safeSearch": "moderate",
    }

    try:
        resp = requests.get(endpoint, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        # Don’t crash the app — just return empty so app.py can fallback
        return []

    results: List[Dict] = []
    for item in data.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        sn = item.get("snippet") or {}
        if not vid:
            continue

        thumbs = sn.get("thumbnails") or {}
        thumb = (thumbs.get("medium") or thumbs.get("high") or thumbs.get("default") or {}).get("url")

        results.append(
            {
                "title": sn.get("title", "Untitled"),
                "channel": sn.get("channelTitle", ""),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "thumbnail": thumb,
            }
        )

    return _dedupe_by_url(results)