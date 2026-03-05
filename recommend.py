from __future__ import annotations

from typing import Dict, List
import re
import requests

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "you", "we", "they", "i", "he", "she", "them", "our", "your", "my", "at", "by",
    "from", "not", "do", "does", "did", "can", "could", "should", "would", "will", "just",
    "about", "into", "over", "than", "also"
}


def _top_keywords(text: str, k: int = 10) -> List[str]:
    text = (text or "").lower()
    words = re.findall(r"[a-zA-Z]{4,}", text)

    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1

    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:k]
    return [w for w, _ in top]


def fallback_search_links(transcript_text: str, video_url: str) -> List[Dict]:
    kws = _top_keywords(transcript_text, k=10)

    query1 = "+".join(kws[:6]) if kws else "tutorial"
    query2 = "+".join(kws[:3]) if kws else "learning"

    return [
        {
            "title": "Search similar videos (based on transcript keywords)",
            "channel": "YouTube Search",
            "url": f"https://www.youtube.com/results?search_query={query1}",
            "thumbnail": None,
        },
        {
            "title": "More videos on this topic (broader search)",
            "channel": "YouTube Search",
            "url": f"https://www.youtube.com/results?search_query={query2}",
            "thumbnail": None,
        },
        {
            "title": "Open the current video",
            "channel": "Current",
            "url": video_url,
            "thumbnail": None,
        },
    ]


def related_videos_youtube_api(
    youtube_api_key: str,
    transcript_text: str,
    max_results: int = 8,
    region_code: str = "IN",
    relevance_language: str = "en",
) -> List[Dict]:
    if not youtube_api_key:
        return []

    kws = _top_keywords(transcript_text, k=10)
    q = " ".join(kws[:6]) if kws else "tutorial"

    endpoint = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "type": "video",
        "maxResults": max_results,
        "q": q,
        "key": youtube_api_key,
        "regionCode": region_code,
        "relevanceLanguage": relevance_language,
        "safeSearch": "moderate",
    }

    resp = requests.get(endpoint, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    results: List[Dict] = []
    for item in data.get("items", []):
        vid = (item.get("id") or {}).get("videoId")
        sn = item.get("snippet") or {}
        if not vid:
            continue

        thumbs = sn.get("thumbnails") or {}
        thumb = (thumbs.get("medium") or thumbs.get("default") or {}).get("url")

        results.append(
            {
                "title": sn.get("title", "Untitled"),
                "channel": sn.get("channelTitle", ""),
                "url": f"https://www.youtube.com/watch?v={vid}",
                "thumbnail": thumb,
            }
        )

    return results