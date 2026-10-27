"""Content deduplication, fingerprinting, and redirect collapsing (Task 3.3)."""

import hashlib
import json
import re
from collections.abc import Sequence

from pydantic import TypeAdapter

from lore_mcp.domain.documents import NormalizedDocument
from lore_mcp.domain.normalization import DeduplicationReport
from lore_mcp.ports.normalization import Deduplicator

REDIRECT_PATTERN = re.compile(
    r"^(?:#REDIRECT|REDIRECT)\s*(?:\[\[\s*)?([^\]\r\n]+?)(?:\s*\]\])?$",
    re.IGNORECASE,
)


def compute_normalized_hash(content: str) -> str:
    """Compute deterministic SHA-256 over normalized whitespace-collapsed text."""
    # Collapse whitespace and lowercase
    cleaned = re.sub(r"\s+", " ", content.strip().lower())
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def extract_token_shingles(text: str, k: int = 3) -> set[str]:
    """Extract set of word k-grams (shingles) for similarity comparison."""
    words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Calculate Jaccard similarity coefficient between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0


class ContentDeduplicator(Deduplicator):
    """Detects duplicates, collapses redirects, and eliminates repeated content."""

    def __init__(self, near_duplicate_threshold: float = 0.92) -> None:
        self._near_duplicate_threshold = near_duplicate_threshold
        self._seen_hashes: set[str] = set()

    def compute_fingerprint(self, content: str) -> str:
        """Compute a deterministic hash fingerprint for content comparison."""
        return compute_normalized_hash(content)

    def is_redirect_document(self, doc: NormalizedDocument) -> str | None:
        """Return the target article title if document is a pure redirect, else None."""
        # 1. Check metadata
        target = doc.metadata.get("redirect_target")
        if target:
            return target.strip()

        # 2. Check document text content
        match = REDIRECT_PATTERN.match(doc.text.strip())
        if match:
            return match.group(1).strip()

        return None

    def filter_duplicates(
        self,
        docs: Sequence[NormalizedDocument],
    ) -> tuple[list[NormalizedDocument], DeduplicationReport]:
        """Eliminate redundant documents and collapse redirects into target docs."""
        if not docs:
            return [], DeduplicationReport()

        exact_dropped = 0
        redirects_collapsed = 0
        recap_skipped = 0

        # Step 1: Index canonical documents by title and URL
        # title_lower -> doc
        title_to_doc: dict[str, NormalizedDocument] = {}
        for doc in docs:
            # If doc is not a redirect itself, record title
            if not self.is_redirect_document(doc):
                title_to_doc[doc.title.strip().lower()] = doc

        # Step 2: Identify redirects and map aliases (title_lower -> aliases)
        redirect_map: dict[str, list[str]] = {}

        for doc in docs:
            target_title = self.is_redirect_document(doc)
            if target_title:
                t_lower = target_title.lower()
                redirect_map.setdefault(t_lower, []).append(doc.title)
            # Also check metadata redirects from MediaWiki
            meta_redirects_raw = doc.metadata.get("redirects")
            if meta_redirects_raw:
                try:
                    adapter = TypeAdapter(list[str])
                    meta_list = adapter.validate_json(meta_redirects_raw)
                    for alias in meta_list:
                        redirect_map.setdefault(doc.title.strip().lower(), []).append(
                            alias
                        )
                    redirects_collapsed += len(meta_list)
                except Exception:
                    pass

        # Step 3: Process documents, collapse redirects, and filter duplicates
        output_docs: list[NormalizedDocument] = []
        batch_seen_fingerprints: set[str] = set(self._seen_hashes)
        # For near-duplicate transcript detection: fingerprint -> doc_id
        transcript_shingles: list[tuple[NormalizedDocument, set[str]]] = []

        for doc in docs:
            # If document is a pure redirect, collapse it and skip
            if self.is_redirect_document(doc):
                redirects_collapsed += 1
                continue

            # Exact fingerprint check
            fingerprint = self.compute_fingerprint(doc.text)
            if fingerprint in batch_seen_fingerprints:
                exact_dropped += 1
                continue

            # Transcript-specific near duplicate / recap check
            if doc.source_type == "transcript":
                current_shingles = extract_token_shingles(doc.text, k=4)
                is_duplicate_transcript = False
                for prior_doc, prior_shingles in transcript_shingles:
                    # Same season & episode: duplicate subtitle release
                    same_episode = doc.metadata.get("season") == prior_doc.metadata.get(
                        "season"
                    ) and doc.metadata.get("episode") == prior_doc.metadata.get(
                        "episode"
                    )
                    similarity = jaccard_similarity(current_shingles, prior_shingles)
                    if same_episode and similarity >= self._near_duplicate_threshold:
                        is_duplicate_transcript = True
                        exact_dropped += 1
                        break
                    elif (
                        not same_episode
                        and similarity >= self._near_duplicate_threshold
                    ):
                        # Potential duplicate episode upload under wrong episode number
                        is_duplicate_transcript = True
                        recap_skipped += 1
                        break

                if is_duplicate_transcript:
                    continue
                transcript_shingles.append((doc, current_shingles))

            # Merge collected redirect aliases into doc metadata
            t_lower = doc.title.strip().lower()
            aliases = sorted(set(redirect_map.get(t_lower, [])))
            updated_meta = dict(doc.metadata)
            if aliases:
                updated_meta["aliases"] = json.dumps(aliases)
                updated_meta["redirects"] = json.dumps(aliases)

            updated_doc = doc.model_copy(update={"metadata": updated_meta})
            batch_seen_fingerprints.add(fingerprint)
            self._seen_hashes.add(fingerprint)
            output_docs.append(updated_doc)

        report = DeduplicationReport(
            total_documents_processed=len(docs),
            unique_documents_output=len(output_docs),
            exact_duplicates_dropped=exact_dropped,
            redirects_collapsed=redirects_collapsed,
            recap_cues_skipped=recap_skipped,
        )

        return output_docs, report
