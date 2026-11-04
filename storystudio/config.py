"""Settings, all from environment variables (see compose.yaml)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


@dataclass(frozen=True)
class Settings:
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    story_model: str = field(default_factory=lambda: _env("STORY_MODEL", "gemma4:12b"))
    image_model: str = field(default_factory=lambda: _env("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B"))
    image_revision: str = field(
        default_factory=lambda: _env("IMAGE_MODEL_REVISION", "e7b7dc27f91deacad38e78976d1f2b499d76a294")
    )
    stories_dir: Path = field(default_factory=lambda: Path(_env("STORIES_DIR", "stories")))

    # Story model sampling. num_ctx fits a whole story + critique comfortably.
    temperature: float = 0.8
    critic_temperature: float = 0.2
    num_ctx: int = 16384
    max_refine_rounds: int = 3

    # FLUX.2 [klein] 4B is step-distilled: 4 steps, guidance 1.0.
    image_steps: int = 4
    image_guidance: float = 1.0
    page_size: tuple[int, int] = (1024, 768)  # width, height (landscape picture book)
    ref_size: tuple[int, int] = (768, 1024)  # character reference, portrait
    max_refs_per_page: int = 5


settings = Settings()
