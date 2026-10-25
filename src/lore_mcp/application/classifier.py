"""Query classifier and alias expansion service (Tasks 8.1 & 8.2)."""

import re
from dataclasses import dataclass, field
from typing import Literal

from lore_mcp.domain.entities import AliasTable, SeriesManifest

QueryClass = Literal["simple_factual", "entity_focused", "complex_multi_entity"]
QueryIntent = Literal["dialogue", "event", "character", "general"]

# Complex / multi-entity pattern indicators
_COMPLEX_COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bdifference between\b",
    r"\bdifferences between\b",
    r"\brelationship between\b",
    r"\bvs\.?\b",
    r"\bversus\b",
]

_COMPLEX_COMPOUND_PATTERNS = [
    r"\band (?:why|what|how|where|when|then|afterwards|later)\b",
    r"\bas well as\b",
]

# Simple factual inquiry patterns
_FACTUAL_PATTERNS = [
    r"\bwhen (?:was|is|did|does|will)\b",
    r"\bwhere (?:was|is|did|does|can)\b",
    r"\bwhat (?:episode|season|year|date|time)\b",
    r"\bwhich (?:episode|season|year)\b",
    r"\bhow many\b",
    r"\bhow old\b",
    r"\bhow long\b",
    r"\bis (?:he|she|it|they|\w+) (?:a|an|the)\b",
    r"\bwhat is the name of\b",
]

# Entity-focused / profiling patterns
_ENTITY_FOCUSED_PATTERNS = [
    r"\bwho (?:is|was|are|were)\b",
    r"\btell me about\b",
    r"\bbackground on\b",
    r"\babilities of\b",
    r"\bpowers? of\b",
    r"\borigin of\b",
    r"\bhistory of\b",
    r"\bprofile of\b",
    r"\bbiograph(?:y|ical)\b",
]

# Common dialogue marker keywords (preserved for intent)
_DIALOGUE_PATTERNS = [
    r"\bwho said\b",
    r"\bsays\b",
    r"\bsaid\b",
    r"\bquote\b",
    r"\bdialogue\b",
    r"\blines?\b",
    r"\bspeech\b",
]

# Common event marker keywords (preserved for intent)
_EVENT_PATTERNS = [
    r"\bwhen did\b",
    r"\bwhat happened\b",
    r"\bhow did\b",
    r"\bbattle\b",
    r"\bfight\b",
    r"\bdeath\b",
    r"\bdies\b",
    r"\bkilled\b",
    r"\bexplod\w*\b",
    r"\bdefeat\w*\b",
]

# Common character / profile marker keywords (preserved for intent)
_CHARACTER_PATTERNS = [
    r"\bwho is\b",
    r"\bwho was\b",
    r"\bpowers?\b",
    r"\babilities\b",
    r"\bbackground\b",
    r"\bbio\w*\b",
    r"\borigin\b",
    r"\bweapon\b",
]


@dataclass(frozen=True)
class ClassifiedQuery:
    """Classified query text with expanded search aliases and detected intent."""

    raw_query: str
    expanded_query: str
    intent: QueryIntent
    matched_aliases: list[str]
    sparse_query: str = ""
    query_class: QueryClass = "simple_factual"
    detected_entities: list[str] = field(default_factory=list[str])


class QueryClassifier:
    """Classifies user queries and expands entity/series aliases for search."""

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the user query intent based on lexical indicators."""
        q_lower = query.lower().strip()

        for pat in _DIALOGUE_PATTERNS:
            if re.search(pat, q_lower):
                return "dialogue"

        for pat in _EVENT_PATTERNS:
            if re.search(pat, q_lower):
                return "event"

        for pat in _CHARACTER_PATTERNS:
            if re.search(pat, q_lower):
                return "character"

        return "general"

    def classify(self, query: str) -> QueryIntent:
        """Backward-compatible intent classifier."""
        return self.classify_intent(query)

    def classify_class(
        self,
        query: str,
        detected_entities: list[str] | None = None,
    ) -> QueryClass:
        """Classify query into simple_factual, entity_focused, or complex."""
        q_lower = query.lower().strip()
        entities = detected_entities or []

        # 1. Explicit Complex / Multi-entity conditions
        # a) Comparative phrasing
        for pat in _COMPLEX_COMPARISON_PATTERNS:
            if re.search(pat, q_lower):
                return "complex_multi_entity"

        # b) Compound causal inquiries or multi-sentence questions
        for pat in _COMPLEX_COMPOUND_PATTERNS:
            if re.search(pat, q_lower):
                return "complex_multi_entity"

        if re.search(r"\?.*\?", q_lower) or ";" in q_lower:
            return "complex_multi_entity"

        # 2. Simple factual conditions (e.g. When was X arrested by Y?)
        for pat in _FACTUAL_PATTERNS:
            if re.search(pat, q_lower):
                return "simple_factual"

        # 3. 2 or more distinct entities detected without explicit factual question
        if len(set(entities)) >= 2:
            return "complex_multi_entity"

        # 4. Entity-focused conditions
        if len(set(entities)) == 1:
            for pat in _ENTITY_FOCUSED_PATTERNS:
                if re.search(pat, q_lower):
                    return "entity_focused"
            if len(q_lower.split()) <= 4:
                return "entity_focused"

        # 5. Fallback
        if len(set(entities)) == 1:
            return "entity_focused"

        return "simple_factual"

    def expand_aliases(
        self,
        query: str,
        manifest: SeriesManifest | None = None,
        alias_table: AliasTable | None = None,
    ) -> ClassifiedQuery:
        """Classify query and produce sparse_query with canonical entity terms."""
        intent = self.classify_intent(query)
        q_raw = query.strip()
        q_lower = q_raw.lower()

        matched_aliases: list[str] = []
        detected_entities: list[str] = []
        canonical_terms_to_add: list[str] = []

        alias_map: dict[str, list[str]] = {}

        if manifest:
            for alias, canonical in manifest.entity_aliases.items():
                clean_alias = alias.strip().lower()
                clean_canon = canonical.strip().lower()
                if clean_alias:
                    if clean_alias not in alias_map:
                        alias_map[clean_alias] = []
                    if canonical not in alias_map[clean_alias]:
                        alias_map[clean_alias].append(canonical)
                if clean_canon:
                    if clean_canon not in alias_map:
                        alias_map[clean_canon] = []
                    if canonical not in alias_map[clean_canon]:
                        alias_map[clean_canon].append(canonical)

            for s_alias in manifest.aliases:
                clean_s = s_alias.strip().lower()
                if clean_s:
                    if clean_s not in alias_map:
                        alias_map[clean_s] = []
                    if manifest.display_name not in alias_map[clean_s]:
                        alias_map[clean_s].append(manifest.display_name)

            clean_disp = manifest.display_name.strip().lower()
            if clean_disp:
                if clean_disp not in alias_map:
                    alias_map[clean_disp] = []
                if manifest.display_name not in alias_map[clean_disp]:
                    alias_map[clean_disp].append(manifest.display_name)

        if alias_table:
            for entry in alias_table.entries:
                clean_k = entry.lookup_key.strip().lower()
                clean_c = entry.canonical_name.strip().lower()
                if clean_k:
                    if clean_k not in alias_map:
                        alias_map[clean_k] = []
                    if entry.canonical_name not in alias_map[clean_k]:
                        alias_map[clean_k].append(entry.canonical_name)
                if clean_c:
                    if clean_c not in alias_map:
                        alias_map[clean_c] = []
                    if entry.canonical_name not in alias_map[clean_c]:
                        alias_map[clean_c].append(entry.canonical_name)

        for alias, canon_list in alias_map.items():
            escaped = re.escape(alias)
            pattern = rf"\b{escaped}\b"
            if re.search(pattern, q_lower):
                for canon in canon_list:
                    if canon not in detected_entities:
                        detected_entities.append(canon)
                    if alias != canon.lower() and alias not in matched_aliases:
                        matched_aliases.append(alias)
                    if (
                        re.search(rf"\b{re.escape(canon.lower())}\b", q_lower)
                        is None
                    ):
                        if canon not in canonical_terms_to_add:
                            canonical_terms_to_add.append(canon)

        query_class = self.classify_class(q_raw, detected_entities)

        if canonical_terms_to_add:
            sparse_query = f"{q_raw} {' '.join(canonical_terms_to_add)}"
        else:
            sparse_query = q_raw

        expanded_query = sparse_query

        return ClassifiedQuery(
            raw_query=q_raw,
            expanded_query=expanded_query,
            intent=intent,
            matched_aliases=matched_aliases,
            sparse_query=sparse_query,
            query_class=query_class,
            detected_entities=detected_entities,
        )
