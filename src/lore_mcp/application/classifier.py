"""Query classifier and alias expansion service."""

import re
from dataclasses import dataclass
from typing import Literal

from lore_mcp.domain.entities import SeriesManifest

QueryIntent = Literal["dialogue", "event", "character", "general"]

# Common dialogue marker keywords
_DIALOGUE_PATTERNS = [
    r"\bwho said\b",
    r"\bsays\b",
    r"\bsaid\b",
    r"\bquote\b",
    r"\bdialogue\b",
    r"\blines?\b",
    r"\bspeech\b",
]

# Common event marker keywords
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

# Common character / profile marker keywords
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


class QueryClassifier:
    """Classifies user queries and expands entity/series aliases for search."""

    def classify(self, query: str) -> QueryIntent:
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

    def expand_aliases(
        self,
        query: str,
        manifest: SeriesManifest | None = None,
    ) -> ClassifiedQuery:
        """Classify query and augment query text with known aliases if applicable."""
        intent = self.classify(query)
        matched_aliases: list[str] = []
        expanded = query.strip()

        if manifest and manifest.aliases:
            q_lower = query.lower()
            # If query mentions a series alias or canonical display name, record it
            for alias in manifest.aliases:
                if alias.lower() in q_lower:
                    matched_aliases.append(alias)

        return ClassifiedQuery(
            raw_query=query,
            expanded_query=expanded,
            intent=intent,
            matched_aliases=matched_aliases,
        )
