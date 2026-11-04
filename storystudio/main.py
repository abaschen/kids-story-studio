"""Web server: the UI, a small JSON API, and the story folders as static files."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .images import FluxKlein, vram_used_mb
from .llm import Ollama
from .store import Store
from .workflow import Workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

store = Store(settings.stories_dir)
workflow = Workflow(
    store,
    Ollama(settings.ollama_url, settings.story_model, settings.num_ctx),
    FluxKlein(settings.image_model, settings.image_revision),
    settings,
)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # A restart drops the in-memory queue: mark leftover jobs so the UI offers to retry them.
    for s in store.summaries():
        if s["state"] in ("queued", "running"):
            store.status(s["id"], state="error", error="interrupted (server restarted)")
    workflow.start()
    yield


app = FastAPI(title="Kids Story Studio", lifespan=lifespan)
app.mount("/files", StaticFiles(directory=settings.stories_dir), name="files")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


def _get(sid: str) -> dict:
    try:
        return store.get(sid)
    except KeyError:
        raise HTTPException(404, "no such story") from None


def _idle(story: dict) -> None:
    if story["status"]["state"] in ("queued", "running"):
        raise HTTPException(409, "this story is busy")


class NewStory(BaseModel):
    context: str = ""
    prompt: str = Field(min_length=10)
    language: str = "English"
    auto: bool = False
    seed: int | None = None


class Feedback(BaseModel):
    feedback: str = Field(min_length=3)


class StoryEdit(BaseModel):
    title: str
    story: str


class Page(BaseModel):
    n: int
    text: str
    illustration: str
    characters: list[str]
    props: list[str] = []
    setting: str = ""
    image: str | None = None
    version: int | None = None


class Render(BaseModel):
    pages: list[int] | None = None
    new_seed: bool = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health() -> dict:
    llm = workflow.llm
    up = llm.reachable()
    return {
        "ollama": up,
        "story_model": settings.story_model,
        "story_model_installed": up and llm.installed(),
        "image_model": f"{settings.image_model}@{settings.image_revision[:10]}",
        "vram_used_mb": vram_used_mb(),
        "queue": workflow.jobs.qsize(),
    }


@app.get("/api/stories")
def list_stories() -> list[dict]:
    return store.summaries()


@app.post("/api/stories")
def create(body: NewStory) -> dict:
    story = store.create(context=body.context, prompt=body.prompt, language=body.language, seed=body.seed)
    workflow.submit(story["id"], "all" if body.auto else "write")
    return store.get(story["id"])


@app.get("/api/stories/{sid}")
def get(sid: str) -> dict:
    return _get(sid)


@app.get("/api/stories/{sid}/calls")
def calls(sid: str) -> list[dict]:
    _get(sid)
    return store.calls(sid)


@app.post("/api/stories/{sid}/run/{kind}")
def run(sid: str, kind: Literal["write", "plan", "render", "all"], body: Render | None = None) -> dict:
    story = _get(sid)
    _idle(story)
    args = body.model_dump() if body and kind == "render" else {}
    workflow.submit(sid, kind, **args)
    return store.get(sid)


@app.post("/api/stories/{sid}/revise")
def revise(sid: str, body: Feedback) -> dict:
    _idle(_get(sid))
    workflow.submit(sid, "revise", feedback=body.feedback)
    return store.get(sid)


@app.put("/api/stories/{sid}/story")
def edit_story(sid: str, body: StoryEdit) -> dict:
    """A hand edit of the text; logged like a model call so the history stays complete."""
    story = _get(sid)
    _idle(story)
    store.log_call(
        sid,
        {
            "kind": "edit",
            "stage": "story",
            "before": {"title": story["title"], "story": story["story"]},
            "after": body.model_dump(),
        },
    )
    return store.update(sid, title=body.title, story=body.story, pages=[], bible=None, refs={})


@app.put("/api/stories/{sid}/pages")
def edit_pages(sid: str, pages: list[Page]) -> dict:
    story = _get(sid)
    _idle(story)
    store.log_call(
        sid, {"kind": "edit", "stage": "pages", "before": story["pages"], "after": [p.model_dump() for p in pages]}
    )
    return store.update(sid, pages=[p.model_dump() for p in pages])
