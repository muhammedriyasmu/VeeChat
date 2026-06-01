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

OVERVIEW_TERMS = {
    "summary", "summarize", "explain", "overview", "key", "points", "main", "idea",
    "ideas", "notes", "important", "takeaways", "takeaway", "theme", "themes",
}

QUERY_EXPANSIONS = {
    "write": {"author", "poet", "written", "writer", "compose", "composed"},
    "writer": {"author", "poet", "write", "written"},
    "author": {"writer", "poet", "write", "written"},
    "poem": {"poetry", "poet", "verse"},
    "explain": {"meaning", "summary", "idea", "theme"},
    "point": {"idea", "topic", "takeaway", "summary"},
    "key": {"main", "important"},
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
            "Answer the question using only the word-matched transcript context below.\n"
            "Rules:\n"
            "1. Do not use outside knowledge.\n"
            "2. If the transcript does not clearly support the answer, say: "
            "\"I could not verify that from the transcript.\"\n"
            "3. If the question asks for explanation, summary, or key points, summarize the matched context.\n"
            "4. Keep the answer accurate, concise, and grounded in the provided context.\n"
            "5. When possible, include a short supporting quote or paraphrase from the transcript.\n\n"
            f"Word-matched transcript context:\n{context}\n\nQuestion: {prompt}"
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

        if q_terms <= OVERVIEW_TERMS:
            return "\n\n".join(c.text for c in self.chunks[:k])

        expanded_terms = self._expand_terms(q_terms)
        matched_sentences = self._word_match_sentences(query, expanded_terms, limit=k * 3)
        if matched_sentences:
            return "\n".join(f"- {sentence}" for sentence in matched_sentences)

        matched_chunks = self._word_match_chunks(query, expanded_terms, limit=k)
        if not matched_chunks:
            if len(q_terms) <= 3:
                return "\n\n".join(c.text for c in self.chunks[:k])
            return ""
        return "\n\n".join(matched_chunks)

    @staticmethod
    def _extract_terms(text: str) -> set[str]:
        words = re.findall(r"[a-zA-Z0-9]{3,}", (text or "").lower())
        return {SimpleRAGBot._normalize_word(word) for word in words if word not in STOPWORDS}

    @staticmethod
    def _normalize_word(word: str) -> str:
        word = word.lower()
        for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
            if len(word) > len(suffix) + 3 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    @staticmethod
    def _expand_terms(terms: set[str]) -> set[str]:
        expanded = set(terms)
        for term in terms:
            expanded.update(SimpleRAGBot._normalize_word(t) for t in QUERY_EXPANSIONS.get(term, set()))
        return expanded

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _word_match_sentences(self, query: str, q_terms: set[str], limit: int = 18) -> list[str]:
        scored: list[tuple[int, int, str]] = []
        query_text = query.lower().strip()

        for chunk in self.chunks:
            for sentence in self._split_sentences(chunk.text):
                sentence_terms = self._extract_terms(sentence)
                overlap = q_terms & sentence_terms
                if not overlap:
                    continue
                phrase_bonus = 3 if query_text and query_text in sentence.lower() else 0
                score = len(overlap) * 2 + phrase_bonus
                scored.append((score, len(overlap), sentence))

        scored.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)

        selected = []
        seen = set()
        for _, _, sentence in scored:
            key = sentence.lower()
            if key in seen:
                continue
            selected.append(sentence)
            seen.add(key)
            if len(selected) >= limit:
                break
        return selected

    def _word_match_chunks(self, query: str, q_terms: set[str], limit: int = 6) -> list[str]:
        scored: list[tuple[int, int, str]] = []
        query_text = query.lower().strip()

        for chunk in self.chunks:
            chunk_terms = self._extract_terms(chunk.text)
            overlap = q_terms & chunk_terms
            overlap_score = len(overlap)
            phrase_bonus = 3 if query_text and query_text in chunk.text.lower() else 0
            score = overlap_score * 2 + phrase_bonus
            scored.append((score, overlap_score, chunk.text))

        scored.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
        return [text for score, overlap_score, text in scored[:limit] if score > 0 and overlap_score > 0]


def make_bot(db_path: str, groq_key: str) -> SimpleRAGBot:
    _ = db_path  # Kept for compatibility with existing app call site.
    return SimpleRAGBot(api_key=groq_key)
