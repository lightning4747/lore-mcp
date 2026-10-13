"""Application service for searching dialogue in transcript chunks."""

import logging
import re

from lore_mcp.application.errors import InvalidArgument
from lore_mcp.application.hybrid import HybridRetriever
from lore_mcp.application.scope import ScopeResolver
from lore_mcp.config import RequestLimitsSettings
from lore_mcp.domain.identifiers import slugify
from lore_mcp.domain.search import DialogueSnippet, SearchHit, SearchQuery

logger = logging.getLogger(__name__)

TIMESTAMP_REGEX = re.compile(
    r"\[?(\d{1,2}:\d{2}:\d{2}(?:[,\.]\d{1,3})?(?:\s*-->\s*\d{1,2}:\d{2}:\d{2}(?:[,\.]\d{1,3})?)?)\]?"
)
SPEAKER_REGEX = re.compile(r"^([A-Z][a-zA-Z0-9_\s'\.]{1,30}):\s*(.*)$")


def _extract_snippet_from_hit(
    hit: SearchHit,
    query_terms: list[str],
    speaker_filter: str | None = None,
) -> DialogueSnippet:
    """Extract a concise dialogue quotation and locators from a transcript chunk."""
    text = hit.chunk.text
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        raw_lines = [text.strip()]

    # Score lines to find the most relevant dialogue line
    best_line = raw_lines[0]
    best_score = -1.0

    for line in raw_lines:
        line_lower = line.lower()
        score = 0.0
        for term in query_terms:
            if term in line_lower:
                score += 5.0
        if speaker_filter and speaker_filter.lower() in line_lower:
            score += 10.0

        if score > best_score:
            best_score = score
            best_line = line

    # 1. Extract timestamp locator
    timestamp_match = TIMESTAMP_REGEX.search(best_line)
    timestamp_str: str | None = None
    cleaned_line = best_line

    if timestamp_match:
        timestamp_str = timestamp_match.group(1).strip()
        cleaned_line = (
            best_line[: timestamp_match.start()] + best_line[timestamp_match.end() :]
        ).strip()

    # 2. Extract speaker locator (only when known)
    speaker_match = SPEAKER_REGEX.match(cleaned_line)
    extracted_speaker: str | None = None
    dialogue_body = cleaned_line

    if speaker_match:
        extracted_speaker = speaker_match.group(1).strip()
        dialogue_body = speaker_match.group(2).strip()
    elif speaker_filter and speaker_filter.lower() in cleaned_line.lower():
        extracted_speaker = speaker_filter
    elif hit.chunk.entity_ids and len(hit.chunk.entity_ids) == 1:
        # If single entity tagged on chunk, use as speaker name candidate
        candidate_slug = str(hit.chunk.entity_ids[0])
        extracted_speaker = candidate_slug.replace("-", " ").title()

    # 3. Form short snippet without verbatim run overrun
    if extracted_speaker and not dialogue_body.startswith(f"{extracted_speaker}:"):
        snippet_content = f"{extracted_speaker}: {dialogue_body}"
    else:
        snippet_content = dialogue_body

    # Cap length to prevent overly long runs while preserving meaning
    if len(snippet_content) > 250:
        snippet_content = snippet_content[:247] + "..."

    return DialogueSnippet(
        content=snippet_content,
        speaker=extracted_speaker,
        season=hit.chunk.season,
        episode=hit.chunk.episode,
        timestamp=timestamp_str,
        score=hit.score,
        chunk_id=hit.chunk.chunk_id,
        provenance=hit.chunk.provenance,
    )


class DialogueService:
    """Orchestrates transcript-only dialogue retrieval with locators."""

    def __init__(
        self,
        scope_resolver: ScopeResolver,
        retriever: HybridRetriever,
        limits_settings: RequestLimitsSettings | None = None,
    ) -> None:
        self._scope_resolver = scope_resolver
        self._retriever = retriever
        self._limits_settings = limits_settings or RequestLimitsSettings()

    async def search_dialogue(
        self,
        query: str,
        series: str,
        season: int | None = None,
        episode: int | None = None,
        speaker: str | None = None,
        entity: str | None = None,
        character: str | None = None,
        max_results: int | None = None,
    ) -> list[DialogueSnippet]:
        """Search transcript chunks only and return short snippets with locators."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise InvalidArgument(
                "Query text must not be empty or whitespace.",
                argument_name="query",
            )
        if len(cleaned_query) > self._limits_settings.max_query_length:
            raise InvalidArgument(
                f"Query exceeds maximum character length of "
                f"{self._limits_settings.max_query_length}.",
                argument_name="query",
            )

        limit = (
            max_results
            if max_results is not None
            else self._limits_settings.default_max_results
        )
        if limit <= 0:
            raise InvalidArgument(
                "max_results must be greater than 0.", argument_name="max_results"
            )
        if limit > self._limits_settings.hard_max_results:
            limit = self._limits_settings.hard_max_results

        # 1. Resolve series scope
        resolved_scope = await self._scope_resolver.resolve(series)
        series_id = resolved_scope.series_id

        # 2. Determine speaker/entity filter
        effective_speaker = (speaker or character or entity or "").strip() or None
        speaker_slug = slugify(effective_speaker) if effective_speaker else None

        # 3. Query vector index restricting to transcript chunks only
        search_query = SearchQuery(
            series_id=series_id,
            query=cleaned_query,
            season=season,
            episode=episode,
            entity=speaker_slug,
            source_type="transcript",
            max_results=limit,
        )
        hits: list[SearchHit] = await self._retriever.retrieve(search_query)

        # Fallback if strict entity tag yielded no hits but speaker name was supplied
        if not hits and effective_speaker:
            augmented_query = f"{effective_speaker} {cleaned_query}"
            fallback_query = SearchQuery(
                series_id=series_id,
                query=augmented_query,
                season=season,
                episode=episode,
                entity=None,
                source_type="transcript",
                max_results=limit,
            )
            hits = await self._retriever.retrieve(fallback_query)

        # 4. Extract short snippets with locators
        query_terms = [t.lower() for t in cleaned_query.split() if len(t) >= 2]
        snippets: list[DialogueSnippet] = []
        for hit in hits:
            text_lower = hit.chunk.text.lower()
            has_term_match = (
                any(term in text_lower for term in query_terms) if query_terms else True
            )
            has_speaker_match = bool(
                effective_speaker and effective_speaker.lower() in text_lower
            )

            # Filter out candidate chunks that have zero term or speaker match
            if query_terms and not has_term_match and not has_speaker_match:
                continue

            snippet = _extract_snippet_from_hit(
                hit=hit,
                query_terms=query_terms,
                speaker_filter=effective_speaker,
            )
            snippets.append(snippet)

        return snippets
