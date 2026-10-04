# System Architecture Overview

The system is a series-agnostic, context-aware Retrieval-Augmented Generation (RAG) backend deployed on Render with docker. It exposes remote lore-retrieval capabilities through the Model Context Protocol (MCP), allowing MCP-compatible clients such as Claude Desktop, Claude Mobile, or other supported LLM interfaces to query detailed, franchise-specific information without exhausting local context limits.

The server performs retrieval only. It contains no LLM: it returns ranked, provenance-tagged evidence, and the MCP client's own LLM handles query decomposition, reasoning, and answer generation.

The system is built around strict universe scoping, hybrid retrieval, entity-aware metadata, narrative chronology, caching, source provenance, and evidence-backed responses.

---

## 1. Data Ingestion and Source Layer

The ingestion system collects information from multiple source tiers to support complex plot, character, event, dialogue, and mechanical-lore questions.

### Source Tiers

**Fandom / MediaWiki Wikis — Primary Depth Source**

Dense entity and lore pages are retrieved through the MediaWiki Action API (`/api.php`) from franchise-specific wiki installations.

Typical content includes:

* Characters
* Locations
* Devices
* Organizations
* Events
* Abilities
* Concepts
* Plot summaries
* Entity relationships

Fandom content is treated as a secondary source rather than unquestionable ground truth. Source provenance is retained for every ingested document and chunk.

**Script / Transcript Repositories — Dialogue Source**

Raw transcripts such as `.srt`, subtitle files, and available script repositories provide exact dialogue and scene-level references.

These sources are useful for:

* Exact statements
* Dialogue context
* Character interactions
* Scene chronology
* Specific terminology

**Wikipedia / MediaWiki Sources — High-Level Source**

Wikipedia and similar MediaWiki APIs provide broader series, season, episode, and character information.

These sources are primarily used for high-level context and metadata.

**Google Search / Tavily — Dynamic Fallback**

External web search is used when the local vector database does not contain sufficient information, particularly for:

* Newly released episodes
* Newly introduced entities
* Recently available information
* Sources not yet processed by the ingestion pipeline

Web results are treated as temporary retrieval context and are not automatically persisted into the primary knowledge base.

All fallback results retain URL, source, retrieval timestamp, and provenance information.

---

## 2. Ingestion and Normalization Pipeline

The ingestion system converts heterogeneous source material into structured, retrievable chunks.

```text
External Sources
      │
      ▼
Source Fetcher
      │
      ▼
Content Normalization
      │
      ▼
Entity / Event Extraction
      │
      ▼
Entity-Aware Chunking
      │
      ▼
Metadata Generation
      │
      ▼
Embedding Generation
      │
      ▼
Qdrant Cloud
```

The relationship between the original source, extracted entity/event, and final vector chunk is preserved through Qdrant payload metadata.

---

## 3. Vector Store and Data Model

The system uses **Qdrant Cloud Free Tier** as the primary persistent vector database.

Qdrant stores both the dense vectors and their associated metadata, eliminating the need for a separate relational database in the initial architecture.

### Metadata Payload

Each indexed chunk contains explicit scope, entity, temporal, and source metadata.

Example:

```json
{
  "series": "loki",
  "entity_type": "device",
  "entity_id": "temporal_loom",
  "primary_concept": "Temporal Loom",
  "related_entities": [
    "Raw Time",
    "Sacred Timeline",
    "He Who Remains",
    "TVA"
  ],
  "season": 2,
  "episode": 6,
  "scene_index": 17,
  "narrative_order": 104,
  "source_type": "fandom",
  "source_url": "...",
  "retrieved_at": "..."
}
```

The temporal metadata distinguishes between:

* **Scene order:** where an event is shown in the episode.
* **Narrative order:** where the event actually occurs in the fictional timeline.

This distinction is important for stories involving time travel, flashbacks, loops, alternate timelines, or non-linear storytelling.

### Embeddings

Source chunks are embedded with a single FastEmbed-compatible text model through Qdrant's inference API, alongside sparse (BM25) vectors for lexical search. Inference can run on Qdrant Cloud or locally via FastEmbed with the same client API.

The model name and vector dimension are recorded in the collection metadata, and the same model is used for both ingestion and queries. Chunks are kept within the model's context window, since over-window input is truncated.

---

## 4. Search and Retrieval Pipeline

The retrieval system combines semantic, lexical, metadata, and chronological information.

```text
User Query
    │
    ▼
Scope Resolution
    │
    ▼
Query Classification (rule-based)
    │
    ├── Simple ───────────────┐
    │                         │
    └── Entity-focused ──────┤
        (alias expansion)     │
                              ▼
                       Hybrid Retrieval
                              │
                   ┌──────────┼──────────┐
                   ▼          ▼          ▼
                 Dense      BM25      Metadata
                 Search     Search     Filters
                   │          │          │
                   └──────────┼──────────┘
                              ▼
                         Rank Fusion
                              │
                              ▼
                          Reranking
                              │
                              ▼
                    Temporal Context Builder
                              │
                              ▼
                       Evidence Builder
```

### Scope Locking

Series scope is a hard retrieval constraint.

For example:

```text
"Loki gets locked in then what happens?"
```

must retrieve only:

```text
series = "loki"
```

The filter is applied directly inside the Qdrant query rather than retrieving unrelated franchises and filtering them afterward.

The retrieval interface therefore requires a series:

```text
search_series_lore(
    query,
    series,
    season?,
    episode?,
    entity?
)
```

---

## 5. Query Classification and Transformation

The server does not use an LLM for query decomposition or rewriting.

Queries are first classified using lightweight, rule-based checks (for example, whether the query contains a known entity or alias).

### Simple Queries

Simple factual questions proceed directly to retrieval.

Example:

```text
"Who is O.B.?"
```

### Entity-Focused Queries

Entity aliases and related terminology are expanded using an alias table built during ingestion.

Example:

```text
"the guy who works at the TVA repair shop"
```

can be resolved toward concepts such as:

```text
O.B.
Temporal Mechanics
TVA
```

### Complex Queries

Queries involving multiple entities, events, causal relationships, or implicit chronology are decomposed by the **MCP client's LLM**, not the server.

The client breaks the question into several focused tool calls.

Example:

```text
"Loki gets locked in then what happens?"
```

may be issued by the client as separate calls:

```text
"Loki trapped in Temporal Loom"
"He Who Remains failsafe"
"Loki time-slipping"
"Loki final Temporal Loom sequence"
```

Each call is a normal, independent retrieval request. This removes server-side LLM latency, cost, and API-key dependencies, and the tool descriptions guide the client toward this decomposition behavior.

---

## 6. Hybrid Retrieval

### Dense Retrieval

Dense vector search handles semantic intent.

For example:

```text
"How does Loki get stuck repeating the same event?"
```

can retrieve information about time slipping or temporal loops even when the exact wording differs.

### BM25 / Sparse Retrieval

Lexical retrieval handles exact terminology and proper nouns such as:

```text
Temporal Loom
O.B.
Raw Time
TVA
He Who Remains
```

### Metadata Filtering

Qdrant payload filters constrain retrieval by:

```text
series
season
episode
entity
entity_type
source_type
```

The series filter is mandatory.

### Rank Fusion

Dense and sparse results are combined using Reciprocal Rank Fusion or an equivalent rank-fusion method.

The resulting candidates are reranked using:

```text
semantic relevance
lexical relevance
entity/concept match
source quality
chronological consistency
```

Chronology is a secondary signal and does not replace relevance ranking.

---

## 7. Entity-Aware Chunking

Documents are not blindly split into arbitrary fixed-size chunks.

The ingestion system attempts to preserve meaningful lore units.

```text
Document
 ├── Entity
 │    ├── Description
 │    ├── Abilities
 │    ├── Relationships
 │    └── History
 │
 └── Events
      ├── Participants
      ├── Location
      ├── Sequence
      └── Consequences
```

Each chunk therefore retains its connection to the underlying entity, event, episode, and source.

This reduces the loss of context caused by arbitrary chunk boundaries.

---

## 8. Temporal Context Construction

Temporal context refers specifically to the distinction between **when an event is shown** and **when the event actually occurs within the fictional timeline**.

For example:

```text
Displayed order:
A → B → C → D

Actual chronological order:
A → C → B → D
```

The retrieval system first determines which evidence is relevant.

The context builder then uses available temporal metadata to organize related evidence into the appropriate narrative order.

```text
Retrieved Evidence
        │
        ▼
Relevant Events
        │
        ▼
Narrative Chronology
        │
        ▼
Chronological Context
        │
        ▼
MCP Client (LLM)
```

This is particularly important for:

* Time travel
* Time loops
* Flashbacks
* Alternate timelines
* Non-linear storytelling

The system does not assume that episode order or scene order represents actual chronological order.

When chronology is uncertain, the system preserves that uncertainty instead of inventing an ordering.

---

## 9. Source Provenance and Evidence

Every retrieved chunk retains its source information.

Example:

```json
{
  "chunk_id": "loki-s2e6-loom-17",
  "source_type": "fandom",
  "source_url": "...",
  "season": 2,
  "episode": 6,
  "scene_index": 17
}
```

The MCP response returns retrieved evidence together with its provenance.

This allows the client LLM's answers to reference:

* Source
* Episode
* Scene
* Relevant entity

Conflicting sources remain distinguishable rather than being silently merged.

---

## 10. Redis Caching Layer

Cloud Run instances are stateless and disposable, so in-process caches are per-instance and lost on cold starts. Redis is the shared, short-lived layer across instances. It has three jobs:

* **Web fallback cache:** conserves limited free search credits.
* **Retrieval result cache:** avoids repeated Qdrant queries when a client re-issues similar requests.
* **Rate limiting:** per-API-key counters shared across instances, protecting search credits.

```text
                    Cloud Run
                       │
                 Query Orchestrator
                       │
                       ▼
                     Redis
                  Cache Lookup
                    /     \
                  HIT      MISS
                  │          │
                  │          ▼
                  │       Retrieval
                  │          │
                  │          ▼
                  │        Qdrant
                  │          │
                  └──────────┘
                       │
                       ▼
                    Response
```

Cache entries:

```text
ret:{series}:{ingest_version}:{hash(query, filters)} → evidence payload   (TTL ~1h)
web:{series}:{hash(query)}                           → filtered fallback results (TTL ~24h)
rate:{api_key}:{window}                              → request counter   (TTL = window)
```

Bumping `ingest_version` after re-ingesting a series invalidates all of its cached retrieval results.

All entries are TTL-bound and rebuildable. A Redis outage degrades to slower responses rather than failures.

---

## 11. Dynamic Web Fallback

If local retrieval does not provide sufficient evidence, the system can invoke external web search.

```text
Local Retrieval
      │
      ▼
Evidence Sufficiency Check
(score thresholds)
      │
   ┌──┴──┐
   │     │
Enough  Insufficient
   │     │
Return   Web Search
Evidence    │
            ▼
     Source Filtering
            │
            ▼
 Temporary Retrieval Context
            │
            ▼
     Evidence Builder
            │
            ▼
      Return Evidence
```

Web results are not automatically inserted into Qdrant.

The fallback layer applies:

* Domain/source filtering
* URL provenance
* Retrieval timestamp
* Duplicate removal
* Series scope validation

---

## 12. MCP Server and Application Layer

The MCP server is implemented using **FastMCP** with **Streamable HTTP transport**.

The Cloud Run service exposes a unified endpoint:

```text
/mcp
```

The MCP layer exposes capabilities rather than internal retrieval mechanisms.

Example tools:

```text
search_series_lore
get_entity
search_dialogue
trace_event
compare_events
```

An MCP-compatible client does not need to know whether the backend uses:

```text
Qdrant
BM25
Redis
web search
```

This keeps the MCP interface stable while allowing the backend implementation to evolve.

The architecture is client-agnostic and is not coupled specifically to Claude Desktop or Claude Mobile.

---

## 13. Cloud Run Infrastructure

The MCP backend is deployed as a stateless Docker container on render.

```text
MCP Client
     │
     │ Streamable HTTP
     ▼
   Render
     │
     ├── FastMCP
     ├── Query Orchestrator
     ├── Retrieval
     ├── Temporal Context Builder
     └── Evidence Builder
            │
            ├── Redis
            ├── Qdrant Cloud
            └── Web Search
```

The container listens on:

```text
0.0.0.0:8080
```

Cloud Run instances are treated as disposable. Persistent data and caching are handled by external services.

---

## 14. End-to-End System Flow

```text
┌──────────────────────┐
│     MCP Client       │
│ Claude / Other LLM   │
└──────────┬───────────┘
           │
           │ Streamable HTTP
           ▼
┌──────────────────────────────────────┐
│          Cloud Run MCP Server        │
│                                      │
│  FastMCP                             │
│     │                                │
│     ▼                                │
│  Scope Resolver                      │
│     │                                │
│     ▼                                │
│  Redis Cache                         │
│     │                                │
│     ├──────── HIT ───────────┐       │
│     │                        │       │
│     └── MISS ────────────────┤       │
│                              ▼       │
│                     Query Classifier │
│                              │       │
│                              ▼       │
│                      Alias Expansion │
│                              │       │
│                              ▼       │
│                     Hybrid Retrieval │
│                              │       │
│                     ┌────────┼───────┐
│                     ▼        ▼       ▼
│                  Qdrant     BM25   Metadata
│                     │        │       │
│                     └────────┼───────┘
│                              │
│                              ▼
│                         Reranking
│                              │
│                              ▼
│                  Temporal Context
│                         Builder
│                              │
│                              ▼
│                     Evidence Builder
└──────────────────────┬───────────────┘
                       │
                       │ Evidence + provenance
                       ▼
                  MCP Client
              (LLM generates answer)
```

---

# Future Extensions

The initial architecture deliberately does not introduce a dedicated knowledge graph into the critical retrieval path.

A future version can add graph-based entity and event relationships once the underlying entity model has stabilized.

Potential relationships include:

```text
Temporal Loom
      │
      ├── affects → Raw Time
      ├── associated_with → TVA
      ├── associated_with → O.B.
      └── created_by → He Who Remains
```

The graph could augment vector retrieval with relationship expansion for complex multi-entity questions.

A dedicated graph database is not required initially; the architecture can first establish the required entity and relationship model before deciding whether a graph-specific storage system is justified.
