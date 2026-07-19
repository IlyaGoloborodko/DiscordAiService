# Build: docker compose build ai     Run: docker compose up -d
#
# Two stages so the shipped image has no build tools in it: the first stage
# installs the dependencies, the second one just runs them.

FROM python:3.14-slim AS builder

# uv, same tool as locally — the lockfile is honoured, so the container gets
# exactly the versions that were tested.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Dependencies first, in their own layer: they change far less often than the
# code, so editing a .py file does not reinstall everything.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev


FROM python:3.14-slim

WORKDIR /app

# Not root: if something ever gets through, it shouldn't own the container.
RUN useradd --create-home --uid 1000 app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app . .

# APP_HOST is 0.0.0.0, not localhost: inside a container localhost means
# "unreachable from outside", and the failure looks like a healthy service
# refusing connections. Reload is a dev convenience and stays off here.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000 \
    APP_RELOAD=false

USER app
EXPOSE 8000

# Migrations run before the server starts, otherwise the very first request
# fails on missing tables. `upgrade head` is a no-op when they are already applied.
#
# The server itself is started through main.py rather than the uvicorn CLI, so
# there is exactly one place that decides how this service runs.
CMD ["sh", "-c", "alembic upgrade head && exec python main.py"]
