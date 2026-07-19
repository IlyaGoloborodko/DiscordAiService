# DiscordAiService

The **AI brain** for a Discord music bot. A FastAPI service that takes a user's
natural-language message ("put on some phonk", "давай на твой выбор", "pause"),
decides what to do, finds real tracks, and returns an answer plus the action the
bot should take. Conversation state is remembered per guild.

The LLM runs locally (via an OpenAI-compatible endpoint such as **LM Studio**) and
does tool-calling through [pydantic-ai](https://ai.pydantic.dev/).

## Architecture

This service is one of three cooperating processes:

```
Discord user ──▶ discordAudio (Go bot)      ──▶ POST /agent ──▶  DiscordAiService (this)
                 captures voice/text,                              decides action, picks tracks,
                 plays audio, does TTS  ◀── tool_calls + reply ◀── talks to the LLM + search svc
                                                                        │
                                                                        ▼
                                                        DsBotSearchService (:9000)
                                                        yt-dlp / Last.fm search, playlists,
                                                        similar, charts, stream resolve
```

- **DiscordAiService** (this repo) — the decision layer. Owns the persona, the
  prompt, conversation memory, and the tool-calling loop.
- **[media-source-service](https://github.com/IlyaGoloborodko/media-source-service)** (`:9000`) — media search microservice (`/search`,
  `/playlist`, `/similar`, `/charts`, `/stream`). Not in this repo.
- **[DiscordMusicBot](https://github.com/IlyaGoloborodko/DiscordMusicBot)** (Go) — the Discord client: joins voice, plays audio, runs TTS,
  resolves stream URLs just-in-time. Not in this repo.

## How `/agent` works

The bot POSTs the user's message (plus optional player state and the list of
actions it supports for this turn). The agent:

1. Loads the guild's recent conversation history (Redis hot cache → Postgres).
2. Injects the current system prompt (persona + rules + the bot's declared actions).
3. Lets the LLM call the **search tools** (`search_tracks`, `get_playlist_tracks`,
   `get_similar_tracks`, `get_top_charts`) to find real tracks.
4. Gets back a flat draft — one `action` + a list of `track_ids` + a reply.
5. Resolves the ids to full tracks (dropping any hallucinated id), maps the action
   to the bot's `tool_calls`, derives a TTS-clean `spoken_answer`, and persists the
   turn.

**Dual mode:** if the request declares `tools`, the response is
`ToolCallResponse` (bot-agnostic tool calls). If not, it falls back to the legacy
`AgentResponse` (`action` + `tracks`).

> **Model note.** The output schema is deliberately **flat** (`action` + a flat
> `track_ids` list, never nested `tracks:[{...}]`) — this was required for weak
> local models whose constrained decoder cannot emit arrays of objects. It also
> lets the service drop invented ids for free. The default model is
> `qwen/qwen3.5-9b`, which handles tool-calling reliably; a `ModelHTTPError` retry
> loop absorbs the occasional engine-side parser hiccup.

## Requirements

- Python **3.14+**
- An OpenAI-compatible LLM endpoint (e.g. **LM Studio**) with a tool-calling model
  loaded (default: `qwen/qwen3.5-9b`)
- **Redis** (Docker compose provided) and **PostgreSQL** for conversation memory
- **DsBotSearchService** running on `:9000`
- [uv](https://docs.astral.sh/uv/) for dependency management

## Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env        # then edit values (see Configuration below)

# 3. Start Redis
docker compose up -d redis

# 4. Create the Postgres database (owner/password must match POSTGRES_DSN)
#    e.g. a database `discord_ai` owned by `myuser`

# 5. Apply migrations
uv run alembic upgrade head

# 6. Load the model in LM Studio (id must match TM_MODEL_NAME)
```

## Run

```bash
uv run python main.py
```

Serves on `http://127.0.0.1:8066` (FastAPI docs at `/docs`).

## Endpoints

| Method | Path            | Purpose                                                        |
|--------|-----------------|----------------------------------------------------------------|
| POST   | `/agent`        | Main entry. Message → reply + `tool_calls` (or legacy action). |
| POST   | `/agent/forget` | Clear a session's stored memory (e.g. a poisoned channel).     |
| POST   | `/prompt`       | Legacy one-shot prompt → search string (pre-agent).            |
| POST   | `/tts`          | Text → audio stream (Wyoming/Piper).                           |

### `/agent` request (essentials)

```jsonc
{
  "session": { "guild_id": "123", "user_name": "Sok" },
  "message": "хочу фонк",
  "context": { "now_playing": "...", "queue_len": 3 },   // optional player state
  "tools": [                                              // omit for legacy mode
    { "name": "play",   "description": "Play one track now", "input_schema": {"required": ["tracks"]} },
    { "name": "enqueue","description": "Add several tracks", "input_schema": {"required": ["tracks"]} },
    { "name": "pause",  "description": "Pause playback" }
  ]
}
```

### `/agent` response (tool-calling mode)

```jsonc
{
  "spoken_answer": "Got it! Spinning some phonk vibes for you now.",  // TTS-clean
  "display_text":  "Got it! 🎧 Spinning some phonk vibes for you now.",
  "clarification": null,
  "tool_calls": [
    { "name": "play", "arguments": { "tracks": [ { "id": "...", "title": "..." } ] } }
  ]
}
```

## Configuration

All via `.env` (see [`.env.example`](.env.example)):

| Variable              | Default                | Meaning                                                             |
|-----------------------|------------------------|---------------------------------------------------------------------|
| `TM_BASE_URL`         | `http://127.0.0.1:1234/v1` | LLM endpoint (OpenAI-compatible).                              |
| `TM_API_KEY`          | `not-needed`           | API key (LM Studio ignores it).                                     |
| `TM_MODEL_NAME`       | `qwen/qwen3.5-9b`      | Model id, must match what's loaded in the endpoint.                 |
| `BOT_LANGUAGE`        | `English`              | Language the bot writes/speaks. **Must match the TTS voice.**       |
| `BOT_NAME`            | `Marina`               | Persona name.                                                       |
| `BOT_NAME_FORMS`      | —                      | Comma-separated forms the user addresses the bot by (declensions).  |
| `SEARCH_SERVICE_URL`  | `http://127.0.0.1:9000`| DsBotSearchService base URL.                                        |
| `POSTGRES_DSN`        | —                      | Postgres DSN (normalised to `postgresql+asyncpg://` internally).    |
| `REDIS_URL`           | `redis://127.0.0.1:6380/0` | Redis hot-cache URL.                                            |
| `HISTORY_TOKEN_LIMIT` | `20000`                | Max tokens of intermediate history kept (system prompt + latest message are never trimmed). |
| `LOG_LEVEL`           | `INFO`                 | Console log level.                                                  |
| `TELEGRAM_BOT_TOKEN`  | —                      | Empty = no Telegram logging. From @BotFather.                       |
| `TELEGRAM_CHAT_ID`    | —                      | Chat to send alerts to (negative id for a group).                   |
| `TELEGRAM_LOG_LEVEL`  | `ERROR`                | Level forwarded to Telegram, independent of `LOG_LEVEL`.            |

Changing `BOT_LANGUAGE`, `BOT_NAME`, or `BOT_NAME_FORMS` needs no code edits — the
prompt is templated and reads them per request. Keep the TTS voice in
`discordAudio` on the same language as `BOT_LANGUAGE`.

## Logging

Configured once in [`app/logging_setup.py`](app/logging_setup.py), called from
`main.py` after `load_dotenv()`. Everything goes to the console at `LOG_LEVEL`;
if a Telegram token and chat id are set, records at `TELEGRAM_LOG_LEVEL` and
above are **also** sent to that chat.

- Telegram sending happens on a **background thread** — a request is never held
  up by a network call to Telegram, and an unreachable Telegram is silent.
- Records from `httpx`/`httpcore`/the logging module itself are never forwarded,
  otherwise sending a message would log, which would send a message, forever.
- Messages are capped at 4000 chars, throttled to one per 3s, and dropped once
  100 are queued — Telegram accepts roughly 20/min per chat.
- `uvicorn.run(..., log_config=None)` keeps uvicorn from installing its own
  handlers, so its errors reach Telegram too.

## Memory

- Per-session key: `guild_id` (or `user_id`, or `global`), namespaced with a
  `_MEMORY_VERSION` prefix so a contract change abandons stale, poisoning history.
- **Redis** is the hot cache; **Postgres** (`agent_sessions` table) is durable.
- History is trimmed by **token budget** (`HISTORY_TOKEN_LIMIT`, `tiktoken`); the
  system prompt and the latest user message are never trimmed.
- Only "clean" turns are persisted, so a bad answer can't poison the next turn.

## Testing

```bash
uv run python -m unittest test_agent_contract -v
```

The agent contract is covered by `test_agent_contract.py` (routing, tool-call
building, TTS cleaning, memory trimming, prompt templating) with no live model.
Other test files (`test_voice.py`, `test_search_client.py`, `wyoming_tests.py`)
exercise audio/search and need the respective services or hardware present.
