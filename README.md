# 📺 Chat with YouTube Videos (RAG + Recommendations)

An AI-powered Streamlit application that lets you **chat with YouTube videos** using
**Retrieval-Augmented Generation (RAG)** and also **recommends related videos** based on
the video transcript.

This project extracts YouTube transcripts, stores them in a vector database (Chroma),
and allows users to ask contextual questions with accurate, grounded answers.

---

## ✨ Features

- ✅ Chat with any YouTube video that has captions
- ✅ Uses **RAG (Retrieval-Augmented Generation)** for accurate answers
- ✅ Stores transcripts in **Chroma Vector DB**
- ✅ **Related video recommendations**
  - With YouTube Data API (best results)
  - Fallback keyword-based YouTube search (no API key needed)
- ✅ Clean Streamlit UI
- ✅ Robust error handling & user feedback
- ✅ Windows-friendly setup

---

## 🧠 Architecture (High Level)
