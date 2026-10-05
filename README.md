# Lore MCP Server

A Retrieval-Augmented Generation (RAG) backend implementing the Model Context Protocol (MCP) to provide series-agnostic, entity-aware lore retrieval for Large Language Models.

## Motivation

Large Language Models frequently hallucinate plot details, confuse character arcs, mix up chronological events, or confidently fabricate lore when discussing complex TV series and franchises. Standard model context windows are insufficient to hold entire franchise histories, and baseline web searches often return superficial or uncontextualized summaries.

This project is built to solve that problem. It exposes a precise lore-retrieval engine via MCP, giving LLMs direct access to structured, provenance-tagged evidence from raw scripts, fandom wikis, and structured episode data. The server handles retrieval and temporal ordering only—no internal LLM generation—forcing the client LLM to ground its answers strictly in returned evidence.

---

## Technical Overview

The backend is deployed as a stateless Docker container exposing a Streamable HTTP MCP endpoint. It pairs dense vector search with sparse lexical search and rigid metadata filtering to query franchise lore without polluting the model's primary context window.

### Key Capabilities

* **Strict Universe Scoping:** Every query requires a hard series lock to prevent cross-franchise evidence contamination.
* **Hybrid Retrieval:** Combines Qdrant dense vector search with sparse (BM25) lexical search and metadata filtering, fused via Reciprocal Rank Fusion.
* **Narrative Chronology vs. Scene Order:** Distinguishes between the order an event is displayed on screen and its actual chronological position in the fictional timeline (essential for time travel, loops, and flashbacks).
* **Entity-Aware Chunking:** Chunks are bounded by narrative units (entities, events, dialogue blocks) rather than arbitrary byte lengths, retaining payload metadata for aliases and relationships.
* **Source Provenance:** Every returned text snippet includes exact source tags (wiki URL, season, episode, scene index) allowing the client LLM to cite its sources explicitly.
* **Shared Caching & Rate Limiting:** A Redis layer caches retrieval payloads, handles dynamic web fallback results, and enforces rate limits.
* **Dynamic Web Fallback:** Fallback search triggers only when local database context is insufficient, keeping results transient rather than permanently polluting the core vector index.

---

## MCP Tools Interface

The server exposes standard MCP tools to any compatible client (Claude Desktop, Claude Mobile, custom LLM interfaces):

* `search_series_lore(query, series, season?, episode?, entity?)`: Primary hybrid search endpoint.
* `get_entity(entity_id, series)`: Retrieves structured metadata, relationships, and history for a specific character, device, location, or concept.
* `search_dialogue(query, series, character?)`: Queries raw transcript and subtitle indices for exact quotes and scene chronology.
* `trace_event(event_id, series)`: Returns the causal and temporal sequence of a specific plot point.
* `compare_events(event_id_a, event_id_b, series)`: Constructs comparative context between two narrative occurrences.

Complex queries requiring multi-step reasoning are intended to be decomposed by the client LLM into multiple discrete tool calls.

---

## System Architecture

```text
┌────────────────────────────────────────────────────────┐
│                      MCP Client                        │
│             (Claude / External LLM)                    │
└──────────────────────────┬─────────────────────────────┘
                           │ Streamable HTTP (/mcp)
                           ▼
┌────────────────────────────────────────────────────────┐
│                   Cloud Run Service                    │
│                                                        │
│  FastMCP Layer                                         │
│     │                                                  │
│     ▼                                                  │
│  Scope Resolver (Enforces `series` parameter)          │
│     │                                                  │
│     ▼                                                  │
│  Redis Cache ─────────── (Cache Hit) ────────────────┐ │
│     │                                                │ │
│  (Cache Miss)                                        │ │
│     │                                                │ │
│     ▼                                                │ │
│  Query Classifier & Alias Expansion                   │ │
│     │                                                │ │
│     ▼                                                │ │
│  Hybrid Search Engine                                │ │
│     ├── Qdrant Cloud (Dense Embeddings)              │ │
│     ├── BM25 (Sparse Lexical Search)                 │ │
│     └── Metadata Payload Filtering                   │ │
│     │                                                │ │
│     ▼                                                │ │
│  Rank Fusion & Reranking                             │ │
│     │                                                │ │
│     ▼                                                │ │
│  Temporal Context Builder                            │ │
│     │                                                │ │
│     ▼                                                │ │
│  Evidence Builder ◄──────────────────────────────────┘ │
└──────────────────────────┬─────────────────────────────┘
                           │ Evidence + Provenance
                           ▼
┌────────────────────────────────────────────────────────┐
│                      MCP Client                        │
│          (Generates grounded response)                 │
└────────────────────────────────────────────────────────┘

```

---

## Ingestion Pipeline

Source material passes through a structured normalization pipeline before vectorization:

1. **Source Fetching:** Ingests raw MediaWiki markup from fandom wikis, `.srt`/script files for dialogue, and Wikipedia for high-level episode structures.
2. **Normalization & Extraction:** Extracts distinct entity boundaries, relationships, scene indices, and fictional narrative ordering.
3. **Payload Tagging:** Attaches explicit metadata (`series`, `season`, `episode`, `scene_index`, `narrative_order`, `source_url`) to each chunk.
4. **Vector Indexing:** Generates dense embeddings alongside sparse BM25 indices inside Qdrant Cloud.

---

# Project is still under construction
