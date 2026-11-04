# Kids Story Studio. Works in bash and PowerShell (every recipe is one docker/uv command).

# List recipes
default:
    @just --list

# Build and start; open http://localhost:8765
up:
    docker compose up -d --build

# Stop (models and stories are kept)
down:
    docker compose down

# Follow the app logs
logs:
    docker compose logs -f app

# Download the story model ahead of time
pull-story-model model="gemma4:12b":
    docker compose exec ollama ollama pull {{model}}

# GPU memory right now
gpu:
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Lint and tests (no GPU needed)
test:
    uv run ruff check .
    uv run ruff format --check .
    uv run pytest -q
