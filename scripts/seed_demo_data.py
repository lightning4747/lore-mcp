"""Seed demo lore data for testing MCP server tools against Qdrant."""

import asyncio
from datetime import UTC, datetime

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


async def seed() -> None:
    settings = get_settings()
    client = create_async_qdrant_client(settings.qdrant)

    print("1. Ensuring collections and payload indexes exist...")
    await QdrantInitializer.ensure_collections_ready(client, settings.qdrant)

    series_reg = QdrantSeriesRegistry(client, settings.qdrant)
    entity_reg = QdrantEntityRegistry(client, settings.qdrant)
    event_reg = QdrantEventRegistry(client, settings.qdrant)
    vector_idx = QdrantVectorIndex(client, settings.qdrant)

    series_id = SeriesId("loki")
    now = datetime.now(UTC)

    # 1. Manifest
    print("2. Registering series manifest 'loki'...")
    manifest = SeriesManifest(
        series_id=series_id,
        display_name="Loki (TV Series)",
        kind="show",
        aliases=["MCU Loki", "Loki Series", "Marvel's Loki"],
        seasons=[1, 2],
        entity_count=5,
        chunk_count=6,
    )
    await series_reg.register_series(manifest)

    # 2. Entities
    print("3. Registering entities...")
    entities = [
        Entity(
            entity_id=EntityId("loki"),
            series_id=series_id,
            name="Loki Laufeyson",
            entity_type="character",
            aliases=["God of Stories", "God of Mischief", "Variant L1130"],
            summary=(
                "The God of Mischief from the Marvel Cinematic Universe who, "
                "after escaping with the Tesseract in 2012, is arrested by the "
                "TVA and ultimately evolves into the God of Stories."
            ),
            related_entities=[
                EntityId("mobius"),
                EntityId("sylvie"),
                EntityId("tva"),
            ],
            metadata={"status": "Ascended God", "affiliation": "TVA"},
        ),
        Entity(
            entity_id=EntityId("mobius"),
            series_id=series_id,
            name="Mobius M. Mobius",
            entity_type="character",
            aliases=["Agent Mobius", "Don"],
            summary=(
                "A TVA agent specializing in investigating dangerous time "
                "criminals. He recruits Variant Loki to hunt Sylvie."
            ),
            related_entities=[EntityId("loki"), EntityId("tva")],
            metadata={"status": "Retired", "affiliation": "TVA"},
        ),
        Entity(
            entity_id=EntityId("sylvie"),
            series_id=series_id,
            name="Sylvie Laufeydottir",
            entity_type="character",
            aliases=["The Variant", "Lady Loki"],
            summary=(
                "A female variant of Loki whose timeline was pruned as a child. "
                "She hid in apocalypses before executing revenge on the TVA."
            ),
            related_entities=[EntityId("loki"), EntityId("he-who-remains")],
            metadata={"status": "Active", "affiliation": "Independent"},
        ),
        Entity(
            entity_id=EntityId("tva"),
            series_id=series_id,
            name="Time Variance Authority",
            entity_type="organization",
            aliases=["TVA", "Minutemen"],
            summary=(
                "A bureaucratic organization created outside of space and time "
                "to protect the Sacred Timeline and prevent multiversal war."
            ),
            related_entities=[EntityId("he-who-remains"), EntityId("mobius")],
            metadata={"headquarters": "Null-Time Zone"},
        ),
        Entity(
            entity_id=EntityId("he-who-remains"),
            series_id=series_id,
            name="He Who Remains",
            entity_type="character",
            aliases=["Kang Variant", "The Creator"],
            summary=(
                "The scientist variant who won the multiversal war and "
                "established the TVA to isolate reality from his variants."
            ),
            related_entities=[
                EntityId("tva"),
                EntityId("sylvie"),
                EntityId("loki"),
            ],
            metadata={"location": "Citadel at the End of Time"},
        ),
    ]
    await entity_reg.register_entities(entities)

    # 3. Events
    print("4. Registering narrative events...")
    events = [
        Event(
            event_id=EventId("loki-s01e01-arrest"),
            series_id=series_id,
            name="Loki's Arrest by the TVA",
            description=(
                "After using the Tesseract to escape the Avengers in 2012, "
                "Loki is apprehended by Hunter B-15 and the Minutemen."
            ),
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=100,
            participants=[EntityId("loki"), EntityId("tva")],
            location="Gobi Desert, Mongolia",
            consequences=[
                "Loki brought to TVA for trial",
                "Reset charge deployed",
            ],
        ),
        Event(
            event_id=EventId("loki-s01e02-roxxcart-ambush"),
            series_id=series_id,
            name="Ambush at Roxxcart Haven Hills",
            description=(
                "Loki and Mobius track Sylvie to a superstore in Haven Hills, "
                "Alabama, during a 2050 hurricane apocalypse. Sylvie sends "
                "reset charges through timelines to bomb the Sacred Timeline."
            ),
            season=1,
            episode=2,
            scene_index=4,
            narrative_order=105,
            participants=[
                EntityId("loki"),
                EntityId("mobius"),
                EntityId("sylvie"),
            ],
            location="Haven Hills, Alabama (2050)",
            consequences=[
                "Sacred Timeline branched uncontrollably",
                "Loki follows Sylvie through portal",
            ],
        ),
        Event(
            event_id=EventId("loki-s01e06-citadel-confrontation"),
            series_id=series_id,
            name="Confrontation at the Citadel at the End of Time",
            description=(
                "Loki and Sylvie breach the Citadel and meet He Who Remains. "
                "Sylvie kills him despite Loki's pleas, shattering the Sacred "
                "Timeline and resurrecting the Multiverse."
            ),
            season=1,
            episode=6,
            scene_index=5,
            narrative_order=120,
            participants=[
                EntityId("loki"),
                EntityId("sylvie"),
                EntityId("he-who-remains"),
            ],
            location="Citadel at the End of Time",
            consequences=[
                "Multiversal war looms",
                "He Who Remains killed",
                "Loki stranded in rewritten TVA",
            ],
        ),
    ]
    for ev in events:
        await event_reg.register_event(ev)

    # 4. Chunks
    print("5. Upserting lore and dialogue chunks...")
    chunks = [
        Chunk(
            chunk_id=ChunkId("chunk-loki-tva-intro"),
            series_id=series_id,
            text=(
                "The Time Variance Authority (TVA) is a sprawling bureaucracy "
                "created by He Who Remains to preserve a single timeline known "
                "as the Sacred Timeline. Minutemen monitor temporal variations "
                "known as Nexus Events and prune branched realities."
            ),
            season=1,
            episode=1,
            scene_index=1,
            narrative_order=101,
            content_hash="hash-chunk-tva-intro",
            entity_ids=[EntityId("tva"), EntityId("he-who-remains")],
            provenance=Provenance(
                source_url="https://marvel.fandom.com/wiki/Time_Variance_Authority",
                source_type="fandom",
                license="CC-BY-SA-3.0",
                attribution_text="Marvel Database Contributors",
                retrieval_timestamp=now,
            ),
        ),
        Chunk(
            chunk_id=ChunkId("chunk-loki-sylvie-backstory"),
            series_id=series_id,
            text=(
                "Sylvie is a Variant of Loki who was targeted for pruning as a "
                "young girl in Asgard. Escaping the TVA with a stolen TemPad, "
                "she spent decades hiding inside catastrophic apocalypses "
                "where temporal variations are hidden by total destruction."
            ),
            season=1,
            episode=2,
            scene_index=3,
            narrative_order=104,
            content_hash="hash-chunk-sylvie-backstory",
            entity_ids=[EntityId("sylvie"), EntityId("loki"), EntityId("tva")],
            provenance=Provenance(
                source_url="https://marvel.fandom.com/wiki/Sylvie_Laufeydottir",
                source_type="fandom",
                license="CC-BY-SA-3.0",
                attribution_text="Marvel Database Contributors",
                retrieval_timestamp=now,
            ),
        ),
        Chunk(
            chunk_id=ChunkId("chunk-dialogue-s01e01-glorious-purpose"),
            series_id=series_id,
            text=(
                "Loki: I am Loki of Asgard, and I am burdened with glorious purpose.\n"
                "Mobius: Yeah, about that... we've got a lot of paperwork to "
                "go through, and glory isn't high on the priority list."
            ),
            season=1,
            episode=1,
            scene_index=2,
            narrative_order=102,
            content_hash="hash-dialogue-glorious-purpose",
            entity_ids=[EntityId("loki"), EntityId("mobius")],
            provenance=Provenance(
                source_url="transcripts/loki/s01e01.srt",
                source_type="transcript",
                attribution_text="Loki S01E01 Transcript",
                retrieval_timestamp=now,
            ),
        ),
        Chunk(
            chunk_id=ChunkId("chunk-dialogue-s01e02-salvation"),
            series_id=series_id,
            text=(
                "Mobius: You think you're the only one who can outsmart fate?\n"
                "Loki: No one bad is ever truly bad, and no one good is ever "
                "truly good."
            ),
            season=1,
            episode=2,
            scene_index=2,
            narrative_order=103,
            content_hash="hash-dialogue-salvation",
            entity_ids=[EntityId("loki"), EntityId("mobius")],
            provenance=Provenance(
                source_url="transcripts/loki/s01e02.srt",
                source_type="transcript",
                attribution_text="Loki S01E02 Transcript",
                retrieval_timestamp=now,
            ),
        ),
        Chunk(
            chunk_id=ChunkId("chunk-dialogue-s01e06-confrontation"),
            series_id=series_id,
            text=(
                "He Who Remains: You kill me, and you open up the timeline to an "
                "infinite number of me, none of whom are as polite or diplomatic.\n"
                "Sylvie: You're a liar.\n"
                "Loki: Sylvie, wait! I've been where you are. I've felt what you feel. "
                "I don't want a throne. I just want you to be okay."
            ),
            season=1,
            episode=6,
            scene_index=5,
            narrative_order=121,
            content_hash="hash-dialogue-citadel-climax",
            entity_ids=[
                EntityId("he-who-remains"),
                EntityId("sylvie"),
                EntityId("loki"),
            ],
            provenance=Provenance(
                source_url="transcripts/loki/s01e06.srt",
                source_type="transcript",
                attribution_text="Loki S01E06 Transcript",
                retrieval_timestamp=now,
            ),
        ),
    ]
    await vector_idx.upsert(chunks)

    print("Successfully seeded all demo data into Qdrant!")
    await client.close()


if __name__ == "__main__":
    asyncio.run(seed())
