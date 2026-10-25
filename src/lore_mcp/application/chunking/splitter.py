"""Chunk size validation and deterministic sentence-boundary splitting (Task 5.4)."""

import re

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


class ChunkSplitter:
    """Enforces context window limits and splits text on sentence boundaries."""

    def __init__(self, max_tokens: int = 256) -> None:
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using WordPiece word ratio (~1.3 tokens/word)."""
        words = text.split()
        if not words:
            return 0
        return round(len(words) * 1.3)

    def split(
        self,
        content: str,
        heading: str = "",
        split_index_offset: int = 0,
    ) -> list[tuple[str, int]]:
        """Split content into deterministic chunks respecting max_tokens.

        Returns a list of (chunk_text, split_index) tuples.
        Prefixes split chunks with heading when provided.
        """
        trimmed_content = content.strip()
        trimmed_heading = heading.strip()

        if not trimmed_content:
            return []

        heading_prefix = f"{trimmed_heading}\n\n" if trimmed_heading else ""
        candidate = f"{heading_prefix}{trimmed_content}"

        if self.estimate_tokens(candidate) <= self.max_tokens:
            return [(candidate, split_index_offset)]

        # Oversized chunk: must split deterministically on sentence boundaries.
        sentences = [
            s.strip()
            for s in SENTENCE_SPLIT_PATTERN.split(trimmed_content)
            if s.strip()
        ]

        if not sentences:
            sentences = [trimmed_content]

        # Calculate budget reserved for heading prefix
        heading_tokens = self.estimate_tokens(heading_prefix)
        content_budget = max(1, self.max_tokens - heading_tokens)

        groups: list[str] = []
        current_sentences: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.estimate_tokens(sentence)
            # If a single sentence exceeds budget on its own, handle it
            if sent_tokens > content_budget and not current_sentences:
                words = sentence.split()
                sub_group: list[str] = []
                sub_tokens = 0
                for w in words:
                    w_tokens = self.estimate_tokens(w)
                    if sub_tokens + w_tokens > content_budget and sub_group:
                        groups.append(" ".join(sub_group))
                        sub_group = [w]
                        sub_tokens = w_tokens
                    else:
                        sub_group.append(w)
                        sub_tokens += w_tokens
                if sub_group:
                    groups.append(" ".join(sub_group))
                continue

            if current_sentences and (current_tokens + sent_tokens > content_budget):
                groups.append(" ".join(current_sentences))
                current_sentences = [sentence]
                current_tokens = sent_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += sent_tokens

        if current_sentences:
            groups.append(" ".join(current_sentences))

        results: list[tuple[str, int]] = []
        for i, group in enumerate(groups):
            chunk_text = f"{heading_prefix}{group}" if heading_prefix else group
            results.append((chunk_text, split_index_offset + i))

        return results
