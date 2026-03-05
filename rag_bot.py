import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class _Chunk:
    text: str


class SimpleRAGBot:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant") -> None:
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model
        self.chunks: list[_Chunk] = []

    def add(self, content: str, data_type: str = "text", metadata=None) -> None:
        if data_type != "text":
            raise ValueError("Only text data_type is supported.")
        self.chunks = [_Chunk(text=c) for c in self._chunk_text(content)]

    def chat(self, prompt: str) -> str:
        if not self.chunks:
            return "No transcript context loaded."

        context = self._retrieve_context(prompt, k=6)
        user_msg = (
            "Use the following transcript context to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {prompt}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You answer questions from YouTube transcript context. If unsure, say so clearly.",
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
        text = " ".join((text or "").split())
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = max(end - overlap, start + 1)
        return chunks

    def _retrieve_context(self, query: str, k: int = 6) -> str:
        q_terms = set(re.findall(r"[a-zA-Z]{3,}", query.lower()))
        if not q_terms:
            return "\n\n".join(c.text for c in self.chunks[:k])

        scored: list[tuple[int, str]] = []
        for c in self.chunks:
            c_terms = set(re.findall(r"[a-zA-Z]{3,}", c.text.lower()))
            score = len(q_terms & c_terms)
            scored.append((score, c.text))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [t for s, t in scored[:k] if s > 0]
        if not top:
            top = [c.text for c in self.chunks[:k]]
        return "\n\n".join(top)


def make_bot(db_path: str, groq_key: str) -> SimpleRAGBot:
    _ = db_path  # Kept for compatibility with existing app call site.
    return SimpleRAGBot(api_key=groq_key)
