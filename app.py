import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_bot import make_bot
from youtube_utils import safe_fetch_transcript
from recommend import related_videos_youtube_api, fallback_search_links

load_dotenv()

st.set_page_config(page_title="VeeChat", page_icon=":tv:", layout="centered")

st.title("VeeChat")
st.caption("Paste a YouTube link -> load transcript -> chat using RAG -> get related video recommendations")

# -------------------------
# Session State
# -------------------------
if "app" not in st.session_state:
    st.session_state.app = None
if "db_path" not in st.session_state:
    st.session_state.db_path = None
if "current_video_url" not in st.session_state:
    st.session_state.current_video_url = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "transcript_loaded" not in st.session_state:
    st.session_state.transcript_loaded = False
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "word_count" not in st.session_state:
    st.session_state.word_count = 0
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "related_videos" not in st.session_state:
    st.session_state.related_videos = []


with st.expander("How it works", expanded=False):
    st.markdown(
        """
1) Enter your Groq API key
2) Paste a YouTube URL (video must have captions)
3) App fetches transcript and stores it in Chroma (RAG)
4) Ask questions
5) Related videos are recommended from transcript keywords (or YouTube API if provided)

**Tip:** If a video has no captions, transcript will fail (this is normal).
        """
    )

default_groq = os.getenv("GROQ_API_KEY", "")
default_yt = os.getenv("YOUTUBE_API_KEY", "")

groq_key = st.text_input("Groq API Key", type="password", value=default_groq)
youtube_api_key = st.text_input(
    "YouTube API Key (optional, for better related videos)",
    type="password",
    value=default_yt,
)

st.divider()

video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")

colA, colB = st.columns([1, 1])
with colA:
    load_btn = st.button("Load Video", use_container_width=True)
with colB:
    clear_btn = st.button("Clear", use_container_width=True)

if clear_btn:
    st.session_state.app = None
    st.session_state.db_path = None
    st.session_state.current_video_url = None
    st.session_state.current_video_id = None
    st.session_state.transcript_loaded = False
    st.session_state.transcript_text = None
    st.session_state.word_count = 0
    st.session_state.chat_history = []
    st.session_state.related_videos = []
    st.rerun()


def _build_new_bot() -> None:
    # Some downstream libs read API keys only from env vars.
    os.environ["GROQ_API_KEY"] = groq_key.strip()
    db_path = tempfile.mkdtemp()
    st.session_state.db_path = db_path
    st.session_state.app = make_bot(db_path, groq_key)


def _precheck_groq() -> tuple[bool, str]:
    key = groq_key.strip()
    if not key:
        return False, "Please enter your Groq API key first."

    try:
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        # Tiny chat call: validates key, quota, and model access.
        client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True, ""
    except Exception as e:
        msg = str(e)
        if "insufficient_quota" in msg or "RateLimitError" in msg:
            return False, "Groq API quota/rate limit exceeded. Check your Groq Cloud usage and try again."
        if "AuthenticationError" in msg or "invalid_api_key" in msg:
            return False, "Invalid Groq API key. Please enter a valid key."
        if "model_not_found" in msg or "does not exist" in msg:
            return False, "Your API key does not have access to the configured model."
        return False, f"Groq pre-check failed: {type(e).__name__}: {e}"


def _compute_related_videos() -> None:
    text = st.session_state.transcript_text or ""
    url = st.session_state.current_video_url or ""

    if youtube_api_key.strip():
        try:
            vids = related_videos_youtube_api(
                youtube_api_key=youtube_api_key.strip(),
                transcript_text=text,
                max_results=8,
                region_code="IN",
                relevance_language="en",
            )
            st.session_state.related_videos = vids if vids else fallback_search_links(text, url)
            return
        except Exception as e:
            st.warning(f"YouTube API related videos failed: {type(e).__name__}: {e}")
            st.session_state.related_videos = fallback_search_links(text, url)
    else:
        st.session_state.related_videos = fallback_search_links(text, url)


if load_btn:
    if not video_url.strip():
        st.error("Please enter a YouTube URL.")
    else:
        with st.spinner("Validating Groq key/quota..."):
            ok, check_msg = _precheck_groq()
        if not ok:
            st.error(check_msg)
            st.stop()

        if video_url != st.session_state.current_video_url or not st.session_state.transcript_loaded:
            with st.spinner("Fetching transcript..."):
                video_id, transcript = safe_fetch_transcript(video_url)

            if transcript.startswith("No transcript"):
                st.error("Cannot load this video: No transcript available.")
                st.session_state.transcript_loaded = False
                st.session_state.current_video_url = None
                st.session_state.current_video_id = None
                st.session_state.transcript_text = None
                st.session_state.word_count = 0
                st.session_state.related_videos = []
            else:
                _build_new_bot()

                with st.spinner("Adding transcript to knowledge base (RAG)..."):
                    st.session_state.app.add(
                        transcript,
                        data_type="text",
                        metadata={"title": "Unknown", "url": video_url, "video_id": video_id},
                    )

                st.session_state.current_video_url = video_url
                st.session_state.current_video_id = video_id
                st.session_state.transcript_loaded = True
                st.session_state.transcript_text = transcript
                st.session_state.word_count = len(transcript.split())
                st.session_state.chat_history = []

                _compute_related_videos()

                st.success("Video loaded. Ask your questions below.")
                st.info(f"Transcript words: {st.session_state.word_count}")
        else:
            st.info("This video is already loaded. Ask your questions below.")


# -------------------------
# UI: Status + Related + Chat
# -------------------------
if st.session_state.transcript_loaded:
    st.success(f"Loaded: {st.session_state.current_video_url}")
    st.caption(
        f"Video ID: {st.session_state.current_video_id} | Transcript words: {st.session_state.word_count}"
    )

    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.subheader("Related Videos")
    with head_col2:
        if st.button("Refresh", key="refresh_related", use_container_width=True):
            _compute_related_videos()
            st.rerun()

    if st.session_state.related_videos:
        for idx, v in enumerate(st.session_state.related_videos):
            title = v.get("title", "Untitled")
            url = v.get("url", "")
            channel = v.get("channel", "")
            thumb = v.get("thumbnail")

            cols = st.columns([1, 4])
            with cols[0]:
                if thumb:
                    st.image(thumb, use_container_width=True)
                else:
                    st.write("No preview")
            with cols[1]:
                st.markdown(f"**{title}**")
                if channel:
                    st.caption(channel)
                if url:
                    st.markdown(f"[Open Video]({url})")
    else:
        st.info("No recommendations yet. Load a video first.")

    st.divider()

    st.subheader("Ask questions")
    prompt = st.text_input("Your question", placeholder="Summarize the video / explain topic / key points...")

    if prompt:
        if st.session_state.app is None:
            st.error("Bot not initialized. Reload the video.")
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = st.session_state.app.chat(prompt)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                if "APIStatusError.__init__()" in err:
                    root = e.__cause__ or e.__context__
                    root_msg = f"{type(root).__name__}: {root}" if root else "Unknown upstream API error"
                    st.error(
                        "Chat failed due to an upstream dependency error while handling an API failure."
                    )
                    st.info(
                        "This usually means your Groq API key is invalid, rate-limited, or does not have access to the configured model."
                    )
                    st.code(f"Underlying error: {root_msg}", language="text")
                    st.info("Re-enter your API key, reload the video, and try the question again.")
                else:
                    st.error(f"Chat failed: {err}")
            else:
                st.session_state.chat_history.append((prompt, answer))
                st.markdown("### Answer")
                st.write(answer)

    if st.session_state.chat_history:
        with st.expander("Chat History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.divider()
else:
    st.info("Enter keys + YouTube URL, then click Load Video.")
