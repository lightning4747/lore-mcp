"""Ingest Loki Season 1 corpus from series/corpus/loki_corpus.json into Qdrant."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lore_mcp.adapters.qdrant.client import create_async_qdrant_client
from lore_mcp.adapters.qdrant.entity_registry import QdrantEntityRegistry
from lore_mcp.adapters.qdrant.event_registry import QdrantEventRegistry
from lore_mcp.adapters.qdrant.index import QdrantVectorIndex
from lore_mcp.adapters.qdrant.initializer import QdrantInitializer
from lore_mcp.adapters.qdrant.registry import QdrantSeriesRegistry
from lore_mcp.config import get_settings
from lore_mcp.domain.documents import Chunk
from lore_mcp.domain.entities import Entity, Event, SeriesManifest
from lore_mcp.domain.provenance import Provenance
from lore_mcp.domain.types import ChunkId, EntityId, EventId, SeriesId


def load_corpus(file_path: Path) -> dict[str, Any]:
    with open(file_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data


async def ingest(corpus_path: Path | None = None) -> None:
    settings = get_settings()
    client = create_async_qdrant_client(settings.qdrant)

    target_path = corpus_path or Path("series/corpus/loki_corpus.json")
    print(f"Loading corpus from {target_path}...")
    corpus = load_corpus(target_path)

    print("Step 1: Ensuring collections and payload indexes exist...")
    await QdrantInitializer.ensure_collections_ready(client, settings.qdrant)

    series_reg = QdrantSeriesRegistry(client, settings.qdrant)
    entity_reg = QdrantEntityRegistry(client, settings.qdrant)
    event_reg = QdrantEventRegistry(client, settings.qdrant)
    vector_idx = QdrantVectorIndex(client, settings.qdrant)

    # 1. Manifest
    manifest_data = corpus["manifest"]
    manifest = SeriesManifest(
        series_id=SeriesId(manifest_data["series_id"]),
        display_name=manifest_data["display_name"],
        kind=manifest_data.get("kind", "show"),
        aliases=manifest_data.get("aliases", []),
        seasons=manifest_data.get("seasons", [1]),
        entity_count=len(corpus.get("entities", [])),
        chunk_count=len(corpus.get("chunks", [])),
    )
    print(f"Step 2: Registering series manifest '{manifest.series_id}'...")
    await series_reg.register_series(manifest)

    # 2. Entities
    entities = [
        Entity(
            entity_id=EntityId(e["entity_id"]),
            series_id=SeriesId(e["series_id"]),
            name=e["name"],
            entity_type=e.get("entity_type", "character"),
            aliases=e.get("aliases", []),
            summary=e.get("summary", ""),
            related_entities=[EntityId(r) for r in e.get("related_entities", [])],
            metadata=e.get("metadata", {}),
        )
        for e in corpus.get("entities", [])
    ]
    print(f"Step 3: Registering {len(entities)} entities in lore_meta...")
    await entity_reg.register_entities(entities)

    # 3. Events
    events = [
        Event(
            event_id=EventId(ev["event_id"]),
            series_id=SeriesId(ev["series_id"]),
            name=ev["name"],
            description=ev.get("description", ""),
            season=ev.get("season"),
            episode=ev.get("episode"),
            scene_index=ev.get("scene_index"),
            narrative_order=ev.get("narrative_order"),
            participants=[EntityId(p) for p in ev.get("participants", [])],
            location=ev.get("location"),
            consequences=ev.get("consequences", []),
        )
        for ev in corpus.get("events", [])
    ]
    print(f"Step 4: Registering {len(events)} narrative events in lore_meta...")
    for ev in events:
        await event_reg.register_event(ev)

    # 4. Chunks
    now = datetime.now(UTC)
    chunks = [
        Chunk(
            chunk_id=ChunkId(c["chunk_id"]),
            series_id=SeriesId(c["series_id"]),
            text=c["text"],
            season=c.get("season"),
            episode=c.get("episode"),
            scene_index=c.get("scene_index"),
            narrative_order=c.get("narrative_order"),
            content_hash=c.get("content_hash", f"hash_{c['chunk_id']}"),
            entity_ids=[EntityId(eid) for eid in c.get("entity_ids", [])],
            provenance=Provenance(
                source_url=c.get("source_url", "https://marvel.fandom.com/wiki/Loki"),
                source_type=c.get("source_type", "fandom"),
                license=c.get("license", "CC-BY-SA-3.0"),
                attribution_text=c.get("attribution_text", "Marvel Wiki Contributors"),
                retrieval_timestamp=now,
            ),
        )
        for c in corpus.get("chunks", [])
    ]
    print(f"Step 5: Upserting {len(chunks)} chunks into lore_chunks...")
    await vector_idx.upsert(chunks)

    # 5. Verification
    print("Step 6: Verifying collection counts...")
    chunks_info = await client.get_collection(settings.qdrant.chunks_collection)
    meta_info = await client.get_collection(settings.qdrant.meta_collection)

    print("Verification complete!")
    print(
        f"  -> '{settings.qdrant.chunks_collection}' points: "
        f"{chunks_info.points_count}"
    )
    print(
        f"  -> '{settings.qdrant.meta_collection}' points: "
        f"{meta_info.points_count}"
    )

    await client.close()


if __name__ == "__main__":
    asyncio.run(ingest())
