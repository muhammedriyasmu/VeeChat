import os
import tempfile
from datetime import datetime
from textwrap import wrap

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_bot import make_bot
from youtube_utils import extract_video_id, safe_fetch_transcript
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
if "chat_summary_text" not in st.session_state:
    st.session_state.chat_summary_text = None
if "chat_summary_pdf" not in st.session_state:
    st.session_state.chat_summary_pdf = None
if "study_notes_text" not in st.session_state:
    st.session_state.study_notes_text = None
if "study_notes_pdf" not in st.session_state:
    st.session_state.study_notes_pdf = None


with st.expander("How it works", expanded=False):
    st.markdown(
        """
1) Paste a YouTube URL (video must have captions)
2) App fetches the transcript and builds transcript chunks for retrieval
3) Ask questions and get Groq answers grounded in the transcript
4) Generate study notes from the transcript for revision
5) Download a PDF summary of the chat session if needed
6) Related videos are recommended from transcript keywords or the YouTube API

**Tip:** If a video has no captions, transcript will fail (this is normal).
        """
    )

default_groq = os.getenv("GROQ_API_KEY", "")
default_yt = os.getenv("YOUTUBE_API_KEY", "")

youtube_api_key = default_yt

st.divider()

video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
manual_transcript = st.text_area(
    "Manual transcript (optional)",
    placeholder="Paste the video transcript here if YouTube blocks automatic caption retrieval.",
    height=140,
)
with st.expander("How to use manual transcript", expanded=False):
    st.markdown(
        """
If YouTube blocks automatic transcript access:

1. Open the video on YouTube.
2. Click the video description / more menu and choose **Show transcript** if available.
3. Copy the transcript text.
4. Paste it into the manual transcript box above.
5. Click **Load Video**.

The app will then answer from your pasted transcript.
        """
    )

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
    st.session_state.chat_summary_text = None
    st.session_state.chat_summary_pdf = None
    st.session_state.study_notes_text = None
    st.session_state.study_notes_pdf = None
    st.rerun()


def _build_new_bot() -> None:
    # Some downstream libs read API keys only from env vars.
    os.environ["GROQ_API_KEY"] = default_groq.strip()
    db_path = tempfile.mkdtemp()
    st.session_state.db_path = db_path
    st.session_state.app = make_bot(db_path, default_groq)


def _precheck_groq() -> tuple[bool, str]:
    key = default_groq.strip()
    if not key:
        return False, "Please add `GROQ_API_KEY` to your project `.env` file first."

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


def _reset_loaded_video_state() -> None:
    st.session_state.transcript_loaded = False
    st.session_state.current_video_url = None
    st.session_state.current_video_id = None
    st.session_state.transcript_text = None
    st.session_state.word_count = 0
    st.session_state.related_videos = []


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


def _load_transcript(video_id: str, transcript: str) -> None:
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
    st.session_state.chat_summary_text = None
    st.session_state.chat_summary_pdf = None
    st.session_state.study_notes_text = None
    st.session_state.study_notes_pdf = None

    _compute_related_videos()

    st.success("Video loaded. Ask your questions below.")
    st.info(f"Transcript words: {st.session_state.word_count}")


def _fallback_chat_summary() -> str:
    history = st.session_state.chat_history
    video_url_value = st.session_state.current_video_url or "Not available"
    video_id_value = st.session_state.current_video_id or "Unknown"

    lines = [
        "VeeChat Session Summary",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Video ID: {video_id_value}",
        f"Video URL: {video_url_value}",
        f"Questions answered: {len(history)}",
        "",
        "Discussion highlights:",
    ]

    for idx, (question, answer) in enumerate(history, 1):
        compact_answer = " ".join((answer or "").split())
        if len(compact_answer) > 280:
            compact_answer = compact_answer[:277] + "..."
        lines.append(f"{idx}. Question: {question}")
        lines.append(f"   Answer: {compact_answer}")

    lines.append("")
    lines.append("Note: This summary used a local fallback because the AI summary request was unavailable.")
    return "\n".join(lines)


def _generate_chat_summary() -> str:
    history = st.session_state.chat_history
    if not history:
        raise ValueError("No chat history available to summarize.")

    transcript_words = st.session_state.word_count
    history_text = "\n\n".join(
        f"Question {idx}: {question}\nAnswer {idx}: {answer}"
        for idx, (question, answer) in enumerate(history, 1)
    )

    prompt = (
        "Create a concise session summary for a user who chatted with a YouTube video.\n"
        "Return plain text only.\n"
        "Use these sections exactly:\n"
        "Title\n"
        "Video Details\n"
        "Questions Covered\n"
        "Key Takeaways\n"
        "Suggested Next Questions\n\n"
        f"Video URL: {st.session_state.current_video_url}\n"
        f"Video ID: {st.session_state.current_video_id}\n"
        f"Transcript word count: {transcript_words}\n"
        f"Chat turns: {len(history)}\n\n"
        f"Chat transcript:\n{history_text}"
    )

    try:
        response = st.session_state.app.client.chat.completions.create(
            model=st.session_state.app.model,
            messages=[
                {
                    "role": "system",
                    "content": "You write clean, useful summaries of chat sessions. Be accurate and concise.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = (response.choices[0].message.content or "").strip()
        return content or _fallback_chat_summary()
    except Exception:
        return _fallback_chat_summary()


def _fallback_study_notes() -> str:
    transcript_text = (st.session_state.transcript_text or "").strip()
    compact_text = " ".join(transcript_text.split())
    excerpt = compact_text[:1800]
    if len(compact_text) > len(excerpt):
        excerpt += "..."

    return "\n".join(
        [
            "Study Notes",
            f"Video ID: {st.session_state.current_video_id or 'Unknown'}",
            f"Video URL: {st.session_state.current_video_url or 'Not available'}",
            "",
            "Overview",
            "These notes are based on the loaded transcript.",
            "",
            "Key Content Excerpt",
            excerpt or "Transcript excerpt not available.",
            "",
            "Review Prompts",
            "- What are the main ideas explained in this video?",
            "- Which examples or steps are most important to remember?",
            "- What would you want to ask next for clarification?",
        ]
    )


def _generate_study_notes() -> str:
    transcript_text = (st.session_state.transcript_text or "").strip()
    if not transcript_text:
        raise ValueError("No transcript available for study notes.")

    transcript_excerpt = transcript_text[:12000]
    prompt = (
        "Create clear study notes from the transcript below.\n"
        "Return plain text only.\n"
        "Use these sections exactly:\n"
        "Title\n"
        "Overview\n"
        "Key Concepts\n"
        "Important Details\n"
        "Quick Review Questions\n"
        "Revision Summary\n\n"
        "Requirements:\n"
        "- Base the notes only on the transcript.\n"
        "- Keep the notes accurate and easy to revise.\n"
        "- Use short bullets where helpful.\n"
        "- Do not invent facts that are not supported by the transcript.\n\n"
        f"Transcript:\n{transcript_excerpt}"
    )

    try:
        response = st.session_state.app.client.chat.completions.create(
            model=st.session_state.app.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create accurate study notes from transcript content. "
                        "Stay grounded in the source and organize the notes clearly."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = (response.choices[0].message.content or "").strip()
        return content or _fallback_study_notes()
    except Exception:
        return _fallback_study_notes()


def _pdf_escape(text: str) -> str:
    sanitized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    sanitized = sanitized.encode("latin-1", "replace").decode("latin-1")
    return sanitized.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_lines(text: str, width: int = 88) -> list[str]:
    wrapped_lines: list[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            wrapped_lines.append("")
            continue
        chunks = wrap(line, width=width, break_long_words=True, break_on_hyphens=False)
        wrapped_lines.extend(chunks or [""])
    return wrapped_lines or [""]


def _build_pdf_bytes(summary_text: str) -> bytes:
    base_lines = _wrap_pdf_lines(summary_text)
    lines_per_page = 44
    page_chunks = [
        base_lines[i:i + lines_per_page] for i in range(0, len(base_lines), lines_per_page)
    ] or [[""]]

    objects: list[bytes] = []

    def _add_object(data: str | bytes) -> int:
        blob = data.encode("latin-1") if isinstance(data, str) else data
        objects.append(blob)
        return len(objects)

    catalog_id = _add_object("<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = _add_object("<< /Type /Pages /Count 0 /Kids [] >>")
    font_id = _add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    for chunk in page_chunks:
        text_ops = ["BT", "/F1 11 Tf", "50 792 Td", "14 TL"]
        for index, line in enumerate(chunk):
            escaped = _pdf_escape(line)
            if index > 0:
                text_ops.append("T*")
            text_ops.append(f"({escaped}) Tj")
        text_ops.append("ET")
        stream = "\n".join(text_ops).encode("latin-1")
        content_id = _add_object(
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        )
        page_id = _add_object(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 842] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Count {len(page_ids)} /Kids [{kids}] >>".encode("latin-1")
    objects[catalog_id - 1] = f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1")

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj_id, obj in enumerate(objects, 1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF"
        ).encode("ascii")
    )
    return bytes(pdf)


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
            pasted_transcript = manual_transcript.strip()

            if pasted_transcript:
                try:
                    video_id = extract_video_id(video_url)
                except Exception as e:
                    st.error(f"Cannot read video URL: {type(e).__name__}: {e}")
                    _reset_loaded_video_state()
                else:
                    _load_transcript(video_id, pasted_transcript)
            else:
                with st.spinner("Fetching transcript..."):
                    transcript_result = safe_fetch_transcript(video_url)

                if transcript_result.ok:
                    _load_transcript(transcript_result.video_id, transcript_result.text)
                else:
                    if transcript_result.status == "blocked":
                        st.warning(transcript_result.message)
                    elif transcript_result.status == "missing":
                        st.error("Cannot load this video: No transcript available.")
                        st.caption(transcript_result.message)
                    else:
                        st.error(transcript_result.message)
                    _reset_loaded_video_state()
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
                    st.info("Update the key in your `.env` file, reload the app, and try again.")
                else:
                    st.error(f"Chat failed: {err}")
            else:
                st.session_state.chat_history.append((prompt, answer))
                st.session_state.chat_summary_text = None
                st.session_state.chat_summary_pdf = None
                st.markdown("### Answer")
                st.write(answer)

    st.divider()

    st.subheader("Study Notes")
    st.caption("Generate revision-ready notes from the loaded transcript.")

    if st.button("Generate Study Notes", use_container_width=True):
        with st.spinner("Preparing study notes..."):
            notes_text = _generate_study_notes()
            st.session_state.study_notes_text = notes_text
            st.session_state.study_notes_pdf = _build_pdf_bytes(notes_text)

    if st.session_state.study_notes_text and st.session_state.study_notes_pdf:
        st.text_area(
            "Study Notes Preview",
            value=st.session_state.study_notes_text,
            height=260,
        )
        st.download_button(
            "Download Study Notes as PDF",
            data=st.session_state.study_notes_pdf,
            file_name=f"veechat-study-notes-{st.session_state.current_video_id or 'session'}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    if st.session_state.chat_history:
        with st.expander("Chat History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.chat_history, 1):
                st.markdown(f"**Q{i}:** {q}")
                st.markdown(f"**A{i}:** {a}")
                st.divider()

        st.subheader("Download Summary")
        st.caption("Create a one-file PDF summary of this chat session.")

        if st.button("Generate Chat Summary PDF", use_container_width=True):
            with st.spinner("Preparing chat summary PDF..."):
                summary_text = _generate_chat_summary()
                st.session_state.chat_summary_text = summary_text
                st.session_state.chat_summary_pdf = _build_pdf_bytes(summary_text)

        if st.session_state.chat_summary_text and st.session_state.chat_summary_pdf:
            st.text_area(
                "Summary Preview",
                value=st.session_state.chat_summary_text,
                height=240,
            )
            st.download_button(
                "Download Summary as PDF",
                data=st.session_state.chat_summary_pdf,
                file_name=f"veechat-summary-{st.session_state.current_video_id or 'session'}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
else:
    st.info("Enter a YouTube URL, then click Load Video.")
