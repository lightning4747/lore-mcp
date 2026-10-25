"""Entity extractor implementing Task 4.1 and Task 4.2."""

import re
from collections.abc import Sequence

from pydantic import TypeAdapter

from lore_mcp.domain.documents import NormalizedDocument
from lore_mcp.domain.entities import Entity, ExtractionReport
from lore_mcp.domain.identifiers import generate_entity_id, slugify
from lore_mcp.domain.series_config import SeriesConfig
from lore_mcp.domain.types import EntityId
from lore_mcp.ports.extraction import EntityExtractor


def _parse_json_list(raw_value: str | None) -> list[str]:
    """Parse JSON string list or return empty list on failure."""
    if not raw_value:
        return []
    try:
        adapter = TypeAdapter(list[str])
        return adapter.validate_json(raw_value)
    except Exception:
        return []


def _extract_summary(text: str, max_length: int = 500) -> str:
    """Extract first non-heading paragraph as synopsis summary."""
    paragraphs = text.split("\n\n")
    for p in paragraphs:
        cleaned = p.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        # Strip internal heading remnants or bullets if any
        first_line = cleaned.split("\n")[0].strip()
        if first_line.startswith("#"):
            continue
        if len(cleaned) <= max_length:
            return cleaned
        truncated = cleaned[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            return truncated[:last_space].rstrip(".,;:") + "..."
        return truncated + "..."
    return ""


class DefaultEntityExtractor(EntityExtractor):
    """Rule-based extractor of typed Entity records from wiki documents."""

    def is_disambiguation_page(
        self,
        title: str,
        categories: Sequence[str],
    ) -> bool:
        """Detect whether a page is a disambiguation page."""
        if title.strip().lower().endswith("(disambiguation)"):
            return True

        for cat in categories:
            clean_cat = cat.lower()
            if "disambiguation" in clean_cat:
                return True

        return False

    def resolve_entity_type(
        self,
        categories: Sequence[str],
        config: SeriesConfig,
    ) -> tuple[str, bool]:
        """Map document categories to an entity type category.

        Returns (entity_type, is_mapped) where is_mapped indicates whether a configured
        category rule matched. If no category matched, defaults to 'concept'.
        """
        # Build normalized lookup for config mappings
        norm_map: dict[str, str] = {
            re.sub(
                r"^category:\s*", "", k.strip(), flags=re.IGNORECASE
            ).lower(): v.strip().lower()
            for k, v in config.category_to_entity_type.items()
        }

        for cat in categories:
            clean_cat = re.sub(
                r"^category:\s*", "", cat.strip(), flags=re.IGNORECASE
            ).lower()
            if clean_cat in norm_map:
                return norm_map[clean_cat], True

        return "concept", False

    def extract_entity(
        self,
        doc: NormalizedDocument,
        config: SeriesConfig,
    ) -> Entity | None:
        """Extract a single Entity from a normalized wiki document."""
        title = doc.metadata.get("title") or doc.title or "Untitled"
        categories = _parse_json_list(doc.metadata.get("categories"))

        if self.is_disambiguation_page(title, categories):
            return None

        entity_id = generate_entity_id(title)
        entity_type, _ = self.resolve_entity_type(categories, config)

        # Collect aliases from redirects and alias metadata
        redirects = _parse_json_list(doc.metadata.get("redirects"))
        explicit_aliases = _parse_json_list(doc.metadata.get("aliases"))
        all_aliases_set: set[str] = set()
        title_lower = title.strip().lower()

        for a in [*redirects, *explicit_aliases]:
            cleaned_alias = a.strip()
            if cleaned_alias and cleaned_alias.lower() != title_lower:
                all_aliases_set.add(cleaned_alias)

        aliases = sorted(all_aliases_set)

        # Collect related entities from internal wiki links
        links = _parse_json_list(doc.metadata.get("links"))
        related_ids: list[EntityId] = []
        seen_related: set[str] = {str(entity_id)}

        for link in links:
            slug = slugify(link)
            if slug and slug not in seen_related:
                seen_related.add(slug)
                related_ids.append(EntityId(slug))

        # Generate summary
        summary = _extract_summary(doc.text)

        # Pass-through metadata
        entity_meta: dict[str, str] = {
            k: v
            for k, v in doc.metadata.items()
            if k not in {"links", "redirects", "aliases", "categories"}
        }
        entity_meta["source_url"] = doc.source_url

        return Entity(
            entity_id=entity_id,
            series_id=config.id,
            name=title,
            entity_type=entity_type,
            aliases=aliases,
            summary=summary,
            related_entities=related_ids,
            metadata=entity_meta,
        )

    def extract_entities(
        self,
        docs: Sequence[NormalizedDocument],
        config: SeriesConfig,
    ) -> tuple[list[Entity], ExtractionReport]:
        """Extract entities across a batch of normalized documents."""
        entities: list[Entity] = []
        disambiguation_skipped = 0
        unmapped_categories_set: set[str] = set()
        unmapped_docs: dict[str, list[str]] = {}

        for doc in docs:
            title = doc.metadata.get("title") or doc.title or "Untitled"
            categories = _parse_json_list(doc.metadata.get("categories"))

            if self.is_disambiguation_page(title, categories):
                disambiguation_skipped += 1
                continue

            _, is_mapped = self.resolve_entity_type(categories, config)

            # Record unmapped categories if document has categories but none matched
            if categories and not is_mapped:
                for cat in categories:
                    clean_cat = re.sub(
                        r"^category:\s*", "", cat.strip(), flags=re.IGNORECASE
                    )
                    unmapped_categories_set.add(clean_cat)
                    unmapped_docs.setdefault(clean_cat, []).append(title)

            entity = self.extract_entity(doc, config)
            if entity is not None:
                entities.append(entity)

        report = ExtractionReport(
            series_id=config.id,
            documents_processed=len(docs),
            entities_extracted=len(entities),
            events_extracted=0,
            disambiguation_pages_skipped=disambiguation_skipped,
            unmapped_categories=sorted(unmapped_categories_set),
            unmapped_category_documents=unmapped_docs,
        )

        return entities, report
