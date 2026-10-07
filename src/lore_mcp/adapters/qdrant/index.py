"""Qdrant VectorIndex adapter implementing chunk indexing [S4, S5, S7-S11]."""

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from lore_mcp.adapters.qdrant.errors import QdrantOperationError
from lore_mcp.config import QdrantSettings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.identifiers import LORE_CHUNK_NAMESPACE
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.search import SearchHit, SearchQuery
from lore_mcp.domain.types import ChunkId, EntityId, SeriesId
from lore_mcp.ports.vector_index import VectorIndex

logger = logging.getLogger(__name__)


def to_point_id(chunk_id: ChunkId | str) -> str:
    """Convert a domain ChunkId into a valid Qdrant point UUID string deterministically.

    If chunk_id is already a valid UUID string (e.g. from generate_chunk_id),
    it is returned as-is. Otherwise, a deterministic UUID5 within LORE_CHUNK_NAMESPACE
    is generated.
    """
    raw_str = str(chunk_id)
    try:
        return str(uuid.UUID(raw_str))
    except ValueError:
        return str(uuid.uuid5(LORE_CHUNK_NAMESPACE, raw_str))


def resolve_embedding_model_name(model_name: str, backend: str) -> str:
    """Normalize dense model name for the target inference backend [S8]."""
    if backend == "fastembed" and "/" not in model_name:
        return f"sentence-transformers/{model_name}"
    return model_name


class QdrantVectorIndex(VectorIndex):
    """Qdrant vector store adapter supporting dense+sparse hybrid indexing."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        settings: QdrantSettings,
    ) -> None:
        self._client: AsyncQdrantClient = client
        self._settings: QdrantSettings = settings
        self._dense_model_name: str = resolve_embedding_model_name(
            model_name=settings.embedding_model,
            backend=settings.inference_backend,
        )

    def _build_point(self, chunk: Chunk) -> models.PointStruct:
        """Construct a Qdrant PointStruct with inference documents and payload."""
        point_id = to_point_id(chunk.chunk_id)
        vectors: dict[str, Any] = {
            self._settings.dense_vector_name: models.Document(
                text=chunk.text,
                model=self._dense_model_name,
            ),
            self._settings.sparse_vector_name: models.Document(
                text=chunk.text,
                model="Qdrant/bm25",
            ),
        }
        payload: dict[str, Any] = {
            "series": str(chunk.series_id),
            "entity_id": [str(e) for e in chunk.entity_ids],
            "entity_type": None,
            "source_type": chunk.provenance.source_type,
            "season": chunk.season,
            "episode": chunk.episode,
            "scene_index": chunk.scene_index,
            "narrative_order": chunk.narrative_order,
            "schema_version": chunk.schema_version,
            "ingest_version": chunk.ingest_version,
            "content_hash": chunk.content_hash,
            "chunk_id": str(chunk.chunk_id),
            "text": chunk.text,
            "source_url": chunk.provenance.source_url,
            "license": chunk.provenance.license,
            "attribution_text": chunk.provenance.attribution_text,
            "revision_id": chunk.provenance.revision_id,
            "retrieval_timestamp": chunk.provenance.retrieval_timestamp.isoformat(),
            "_chunk_json": chunk.model_dump_json(),
        }
        return models.PointStruct(
            id=point_id,
            vector=vectors,
            payload=payload,
        )

    def _reconstruct_chunk_from_payload(
        self, point_id: Any, payload: dict[str, Any]
    ) -> Chunk:
        """Fallback reconstruction of a Chunk if _chunk_json is missing."""
        retrieval_ts: datetime
        if "retrieval_timestamp" in payload and isinstance(
            payload["retrieval_timestamp"], str
        ):
            try:
                retrieval_ts = datetime.fromisoformat(payload["retrieval_timestamp"])
            except ValueError:
                retrieval_ts = datetime.now(UTC)
        else:
            retrieval_ts = datetime.now(UTC)

        raw_chunk_id = str(payload.get("chunk_id", point_id))
        entity_ids = [
            EntityId(str(e))
            for e in payload.get("entity_id", [])
            if isinstance(e, (str, int))
        ]

        return Chunk(
            chunk_id=ChunkId(raw_chunk_id),
            series_id=SeriesId(str(payload.get("series", ""))),
            text=str(payload.get("text", "")),
            season=payload.get("season"),
            episode=payload.get("episode"),
            scene_index=payload.get("scene_index"),
            narrative_order=payload.get("narrative_order"),
            schema_version=int(payload.get("schema_version", 1)),
            ingest_version=int(payload.get("ingest_version", 1)),
            content_hash=str(payload.get("content_hash", "")),
            entity_ids=entity_ids,
            provenance=Provenance(
                source_url=str(payload.get("source_url", "")),
                source_type=str(payload.get("source_type", "")),
                license=payload.get("license"),
                attribution_text=payload.get("attribution_text"),
                retrieval_timestamp=retrieval_ts,
                revision_id=payload.get("revision_id"),
            ),
        )

    async def upsert(self, chunks: Sequence[Chunk]) -> None:
        """Insert or update chunks idempotently in batches."""
        if not chunks:
            return

        batch_size = self._settings.batch_size
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            points = [self._build_point(chunk) for chunk in batch]
            try:
                await self._client.upsert(
                    collection_name=self._settings.chunks_collection,
                    points=points,
                    wait=True,
                )
            except Exception as err:
                raise QdrantOperationError(
                    f"Failed to upsert chunk batch [{i}:{i + len(batch)}] into "
                    f"'{self._settings.chunks_collection}': {err}"
                ) from err

    async def get_by_id(self, chunk_id: ChunkId) -> Chunk | None:
        """Retrieve a specific chunk by its stable identifier."""
        point_id = to_point_id(chunk_id)
        try:
            records = await self._client.retrieve(
                collection_name=self._settings.chunks_collection,
                ids=[point_id],
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to retrieve chunk '{chunk_id}' from "
                f"'{self._settings.chunks_collection}': {err}"
            ) from err

        if not records:
            return None

        record = records[0]
        payload = record.payload or {}
        chunk_json = payload.get("_chunk_json")
        if isinstance(chunk_json, str):
            return Chunk.model_validate_json(chunk_json)

        return self._reconstruct_chunk_from_payload(record.id, payload)

    async def get_existing_ids(self, chunk_ids: Sequence[ChunkId]) -> set[ChunkId]:
        """Return the subset of chunk_ids that already exist in the index."""
        if not chunk_ids:
            return set()

        point_id_to_chunk_id: dict[str, ChunkId] = {
            to_point_id(cid): cid for cid in chunk_ids
        }
        point_ids = list(point_id_to_chunk_id.keys())
        existing: set[ChunkId] = set()

        batch_size = self._settings.batch_size
        for i in range(0, len(point_ids), batch_size):
            batch_ids = point_ids[i : i + batch_size]
            try:
                records = await self._client.retrieve(
                    collection_name=self._settings.chunks_collection,
                    ids=batch_ids,
                    with_payload=False,
                    with_vectors=False,
                )
            except Exception as err:
                raise QdrantOperationError(
                    f"Failed to check existing chunk IDs in "
                    f"'{self._settings.chunks_collection}': {err}"
                ) from err

            for r in records:
                cid = point_id_to_chunk_id.get(str(r.id))
                if cid is not None:
                    existing.add(cid)

        return existing

    async def delete_series(self, series_id: SeriesId) -> None:
        """Delete all indexed chunks for a series."""
        delete_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="series",
                match=models.MatchValue(value=str(series_id)),
            )
        ]
        try:
            await self._client.delete(
                collection_name=self._settings.chunks_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=delete_conditions)
                ),
                wait=True,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to delete chunks for series '{series_id}' from "
                f"'{self._settings.chunks_collection}': {err}"
            ) from err

    async def count(self, series_id: SeriesId | None = None) -> int:
        """Return total chunks count, optionally scoped to a series."""
        count_filter: models.Filter | None = None
        if series_id is not None:
            count_conditions: list[models.Condition] = [
                models.FieldCondition(
                    key="series",
                    match=models.MatchValue(value=str(series_id)),
                )
            ]
            count_filter = models.Filter(must=count_conditions)

        try:
            res = await self._client.count(
                collection_name=self._settings.chunks_collection,
                count_filter=count_filter,
                exact=True,
            )
            return res.count
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to count chunks in '{self._settings.chunks_collection}': {err}"
            ) from err

    async def search(self, query: SearchQuery) -> list[SearchHit]:
        """Search for candidate chunks matching series and constraints."""
        filter_conditions: list[models.Condition] = [
            models.FieldCondition(
                key="series",
                match=models.MatchValue(value=str(query.series_id)),
            )
        ]

        query_filter = models.Filter(must=filter_conditions)

        try:
            response = await self._client.query_points(
                collection_name=self._settings.chunks_collection,
                query=models.Document(
                    text=query.query,
                    model=self._dense_model_name,
                ),
                using=self._settings.dense_vector_name,
                query_filter=query_filter,
                limit=query.max_results,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as err:
            raise QdrantOperationError(
                f"Failed to query chunks in '{self._settings.chunks_collection}': {err}"
            ) from err

        hits: list[SearchHit] = []
        for scored_point in response.points:
            payload = scored_point.payload or {}
            chunk: Chunk | None = None
            chunk_json = payload.get("_chunk_json")
            if isinstance(chunk_json, str):
                chunk = Chunk.model_validate_json(chunk_json)
            else:
                chunk = self._reconstruct_chunk_from_payload(scored_point.id, payload)

            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(scored_point.score),
                    dense_score=float(scored_point.score),
                )
            )
        return hits
