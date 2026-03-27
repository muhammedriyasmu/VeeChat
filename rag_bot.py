import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class _Chunk:
    text: str


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "for",
    "with", "as", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these",
    "those", "you", "we", "they", "i", "he", "she", "them", "our", "your", "my", "at", "by",
    "from", "not", "do", "does", "did", "can", "could", "should", "would", "will", "just",
    "about", "into", "over", "than", "also", "when", "what", "why", "how", "who", "which",
    "where", "there", "their", "have", "has", "had", "more", "most", "very",
}


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
        if not context.strip():
            return (
                "I could not find enough supporting information for that question in the transcript. "
                "Please ask about a topic that is explicitly discussed in the video."
            )

        user_msg = (
            "Answer the question using only the transcript context below.\n"
            "Rules:\n"
            "1. Do not use outside knowledge.\n"
            "2. If the transcript does not clearly support the answer, say: "
            "\"I could not verify that from the transcript.\"\n"
            "3. Keep the answer accurate, concise, and grounded in the provided context.\n"
            "4. When possible, include a short supporting quote or paraphrase from the transcript.\n\n"
            f"Transcript context:\n{context}\n\nQuestion: {prompt}"
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful transcript-grounded assistant. "
                        "Answer only from the provided transcript context and never guess."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
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
        q_terms = self._extract_terms(query)
        if not q_terms:
            return "\n\n".join(c.text for c in self.chunks[:k])

        scored: list[tuple[int, int, str]] = []
        for c in self.chunks:
            c_terms = self._extract_terms(c.text)
            overlap = q_terms & c_terms
            overlap_score = len(overlap)
            phrase_bonus = 3 if query.lower() in c.text.lower() else 0
            score = overlap_score + phrase_bonus
            scored.append((score, overlap_score, c.text))

        scored.sort(key=lambda x: (x[0], x[1], len(x[2])), reverse=True)
        top = [text for score, overlap_score, text in scored[:k] if score > 0 and overlap_score > 0]
        if not top:
            return ""
        return "\n\n".join(top)

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())
        return {word for word in words if word not in STOPWORDS}


def make_bot(db_path: str, groq_key: str) -> SimpleRAGBot:
    _ = db_path  # Kept for compatibility with existing app call site.
    return SimpleRAGBot(api_key=groq_key)
