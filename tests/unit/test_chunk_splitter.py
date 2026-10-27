"""Unit tests for ChunkSplitter and token size validation (Task 5.4)."""

import pytest

from lore_mcp.application.chunking.splitter import ChunkSplitter


@pytest.mark.unit
class TestChunkSplitter:
    """Verifies context window enforcement, sentence splitting, and invariant checks."""

    @pytest.fixture
    def splitter(self) -> ChunkSplitter:
        return ChunkSplitter(max_tokens=50)

    def test_empty_content_returns_empty_list(self, splitter: ChunkSplitter) -> None:
        assert splitter.split("") == []
        assert splitter.split("   ", heading="Section") == []

    def test_short_content_under_limit_preserved(self, splitter: ChunkSplitter) -> None:
        text = "Loki was a Frost Giant adopted by Odin."
        results = splitter.split(content=text, heading="Biography")
        assert len(results) == 1
        chunk_text, split_idx = results[0]
        assert split_idx == 0
        assert chunk_text == "Biography\n\nLoki was a Frost Giant adopted by Odin."

    def test_content_without_heading_returns_content(
        self, splitter: ChunkSplitter
    ) -> None:
        text = "Loki was a Frost Giant."
        results = splitter.split(content=text)
        assert len(results) == 1
        assert results[0] == (text, 0)

    def test_oversized_content_split_on_sentence_boundaries(
        self, splitter: ChunkSplitter
    ) -> None:
        # Each sentence has ~10 words (~13 tokens).
        # max_tokens=50, heading "Origin" is ~2 tokens.
        # 6 sentences = ~60 words, which should result in multiple sub-chunks.
        sentences = [
            f"Sentence number {i} provides detailed historical lore about Asgard."
            for i in range(1, 7)
        ]
        text = " ".join(sentences)

        results = splitter.split(content=text, heading="Origin")
        assert len(results) >= 2

        # Verify each sub-chunk starts with the heading prefix
        for chunk_text, _ in results:
            assert chunk_text.startswith("Origin\n\n")
            tokens = splitter.estimate_tokens(chunk_text)
            assert tokens <= splitter.max_tokens

        # Verify split_index increments sequentially
        indices = [idx for _, idx in results]
        assert indices == list(range(len(results)))

    def test_split_respects_split_index_offset(self, splitter: ChunkSplitter) -> None:
        text = "One short sentence."
        results = splitter.split(content=text, split_index_offset=5)
        assert len(results) == 1
        assert results[0][1] == 5

    def test_single_long_sentence_without_period_does_not_drop_characters(
        self, splitter: ChunkSplitter
    ) -> None:
        # A single sentence with 80 words and no punctuation
        words = [f"word{i}" for i in range(80)]
        text = " ".join(words)

        results = splitter.split(content=text, heading="")
        assert len(results) >= 2

        # Reconstructed words from chunks must equal original words
        reconstructed_words: list[str] = []
        for chunk_text, _ in results:
            reconstructed_words.extend(chunk_text.split())

        assert reconstructed_words == words

    def test_property_boundary_invariants(self) -> None:
        """Property test across various word lengths to verify invariants:
        1. Every word in the source content appears in the chunks.
        2. No chunk exceeds max_tokens.
        """
        custom_splitter = ChunkSplitter(max_tokens=40)
        word_counts = [1, 10, 25, 45, 80, 150]

        for n in word_counts:
            # Generate sentence with n words
            words = [f"token{i}" for i in range(n)]
            content = (
                ". ".join(" ".join(words[i : i + 5]) for i in range(0, n, 5)) + "."
            )
            splits = custom_splitter.split(content=content, heading="Heading")

            assert len(splits) > 0
            for chunk_text, _ in splits:
                assert custom_splitter.estimate_tokens(chunk_text) <= 40
                assert chunk_text.startswith("Heading\n\n")

            # Check all generated words exist in the combined chunk output
            all_text = " ".join(chunk_text for chunk_text, _ in splits)
            for w in words:
                assert w in all_text
