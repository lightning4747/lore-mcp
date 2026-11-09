# Lore MCP: Render Production Deployment Guide

This guide covers deploying `lore-mcp` as a high-performance, stateless server on **Render**, backed by **Qdrant Cloud** (vector store) and **Upstash Redis** (serverless caching and command budgets).

---

## 1. Architecture Overview

```text
┌──────────────────────┐
│  Claude Desktop /    │
│  Claude Mobile       │
└──────────┬───────────┘
           │ Streamable HTTP (HTTPS)
           ▼
┌──────────────────────────────────────┐
│ Render Web Service (lore-mcp)        │
│ • Docker Runtime (Non-root user)     │
│ • Dynamic $PORT listener             │
│ • Touchless /healthz probe           │
└──────────┬───────────────────────────┘
           │
           ├──► Qdrant Cloud (Vector Store & Hybrid Retrieval)
           ├──► Upstash Redis (Serverless Caching & Budget Caps)
           └──► Tavily API (Dynamic Web Search Fallback)
```

Because state, vector indices, and caching are offloaded to managed cloud tiers, the Render container is **100% stateless**. Redeployments and container restarts never lose vector data or cached results.

---

## 2. Prerequisites & Credentials

Gather the following credentials before deploying:

1. **Qdrant Cloud** (Free Tier):
   * Create a free 1GB cluster in the [Qdrant Cloud Console](https://cloud.qdrant.io/).
   * Save your **Cluster URL** (e.g. `https://xxxxxx-xxxx-xxxx.us-east4-0.gcp.cloud.qdrant.io:6333`).
   * Generate and save your **API Key**.

2. **Upstash Redis** (Free Serverless):
   * Create a database in the [Upstash Console](https://console.upstash.com/).
   * In the database details, copy the **`rediss://` URL** (TLS-enabled connection string, e.g. `rediss://default:xxxx@xxxx.upstash.io:6379`).

3. **Tavily API** (Free Search Credits):
   * Obtain an API key from [Tavily](https://tavily.com/) (`tvly-xxxx`).

---

## 3. Deployment Methods

### Option A: 1-Click Render Blueprint (Recommended)

Render Blueprints let you configure the service via [`render.yaml`](../render.yaml):

1. Push your code to your GitHub repository.
2. In the [Render Dashboard](https://dashboard.render.com/), click **New +** → **Blueprint**.
3. Select your repository.
4. Render detects `render.yaml` and prepares the `lore-mcp` service.
5. Provide the required values marked `sync: false`:
   * `QDRANT_URL`: `https://<cluster-id>.cloud.qdrant.io:6333`
   * `QDRANT_API_KEY`: `<your-qdrant-api-key>`
   * `REDIS_URL`: `rediss://default:<password>@<endpoint>.upstash.io:6379`
   * `TAVILY_API_KEY`: `tvly-<your-tavily-key>`
   * `MCP_ACCESS_MODE`: `open` (or `path_token`)
6. Click **Apply**. Render will build and deploy the container.

---

### Option B: Manual Web Service Creation

1. In the [Render Dashboard](https://dashboard.render.com/), click **New +** → **Web Service**.
2. Connect your GitHub repository.
3. Configure the service settings:
   * **Name:** `lore-mcp`
   * **Language / Environment:** `Docker`
   * **Branch:** `main`
   * **Region:** Select the region closest to your Qdrant cluster (e.g. `Oregon (US West)` or `Frankfurt (EU Central)`).
   * **Instance Type:** `Free`
4. Expand **Advanced Settings**:
   * **Health Check Path:** `/healthz`
5. In **Environment Variables**, add the keys defined in [`.env.render.example`](../.env.render.example):
   * `QDRANT_URL`
   * `QDRANT_API_KEY`
   * `REDIS_URL`
   * `TAVILY_API_KEY`
   * `MCP_ACCESS_MODE` = `open`
   * `LOG_LEVEL` = `info`
6. Click **Create Web Service**.

---

## 4. Keeping Free Instances Awake & Active

### Prevent Render 15-Minute Idle Sleep
Render Free Web Services spin down after 15 minutes of zero inbound traffic. Pinging `/healthz` every 10 minutes keeps the server hot:

* **Method 1: GitHub Actions (Built-in)**:
  Configure repository secrets or variables in your GitHub repo:
  * `RENDER_SERVICE_URL`: `https://your-service.onrender.com`
  * `QDRANT_URL`: `https://your-cluster.qdrant.io:6333`
  * `QDRANT_API_KEY`: `your-qdrant-api-key`
  
  The workflow [`.github/workflows/keep_alive.yml`](../.github/workflows/keep_alive.yml) runs every 10 minutes via cron.

* **Method 2: External Cron (e.g. Cron-Job.org / UptimeRobot)**:
  Create a free monitor hitting:
  ```text
  GET https://your-service.onrender.com/healthz
  ```
  Interval: **Every 10 minutes**.

> [!TIP]
> The `/healthz` endpoint is **touchless**—it returns cached application version metadata without contacting Qdrant or Redis. Pinging it 24/7 costs zero Redis or Qdrant quota.

### Prevent Qdrant Cloud 7-Day Cluster Suspension
Free clusters in Qdrant Cloud suspend if inactive for 7 consecutive days. [`scripts/keep_alive.py`](../scripts/keep_alive.py) automatically queries Qdrant `/collections` during scheduled runs to ensure the cluster remains active.

---

## 5. Connecting Claude to Your Lore MCP Server

### Claude Desktop
Add your deployed Render endpoint to your `claude_desktop_config.json`:
* **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "lore": {
      "url": "https://your-service.onrender.com/mcp"
    }
  }
}
```

If you configured `MCP_ACCESS_MODE=path_token` and `MCP_PATH_TOKEN=secret123`:
```json
{
  "mcpServers": {
    "lore": {
      "url": "https://your-service.onrender.com/secret123/mcp"
    }
  }
}
```

### Claude Mobile / Remote Custom Connector
1. Open Claude Settings → **Custom Connectors** / **Add Connector**.
2. Select **Remote Server (SSE / Streamable HTTP)**.
3. Enter `https://your-service.onrender.com/mcp`.
4. Claude will discover tools including `search_series_lore`, `get_entity`, `search_dialogue`, `trace_event`, `compare_events`, and `list_series`.

---

## 6. Post-Deployment Verification

Verify your deployed service from your local terminal using the included smoke test script:

```bash
# Verify container health and tool discovery
python scripts/smoke_test.py --url https://your-service.onrender.com --mode startup

# Run end-to-end verification with a live search query
python scripts/smoke_test.py --url https://your-service.onrender.com --mode deploy
```
