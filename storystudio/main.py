"""Web server: the UI, a small JSON API, and the story folders as static files."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pdf
from .config import settings
from .images import FluxKlein, IdeogramV3, vram_used_mb
from .llm import Ollama
from .store import Store
from .workflow import Workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

store = Store(settings.stories_dir)
# Painters the UI can switch between. Only the selected one is ever loaded.
IMAGE_BACKENDS = {
    "flux-klein-4b": FluxKlein(settings.image_model, settings.image_revision),
    "ideogram-v3": IdeogramV3(settings.ideogram_api_key, settings.ideogram_speed),
}
workflow = Workflow(
    store,
    Ollama(settings.ollama_url, settings.story_model, settings.num_ctx),
    IMAGE_BACKENDS.get(settings.image_backend, IMAGE_BACKENDS["flux-klein-4b"]),
    settings,
)
STATIC = Path(__file__).parent / "static"
PREFS = settings.stories_dir / "_settings.json"
SUGGESTED = ["gemma4:12b", "gemma4:26b-a4b-it-qat", "mistral-small3.2:24b", "qwen3:14b"]


def _load_prefs() -> None:
    """The models picked in the UI survive restarts."""
    try:
        prefs = json.loads(PREFS.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if prefs.get("story_model"):
        workflow.llm.use(prefs["story_model"])
    if prefs.get("image_model") in IMAGE_BACKENDS:
        workflow.images = IMAGE_BACKENDS[prefs["image_model"]]


_load_prefs()


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


def _image_id() -> str:
    return getattr(workflow.images, "id", "")


def _image_options() -> list[dict]:
    out = []
    for key, b in IMAGE_BACKENDS.items():
        usable = not (b.cloud and not settings.ideogram_api_key)
        out.append(
            {
                "id": key,
                "label": b.label,
                "cloud": b.cloud,
                "usable": usable,
                "note": "" if usable else "set IDEOGRAM_API_KEY in .env, then restart",
            }
        )
    return out


@app.get("/api/health")
def health() -> dict:
    llm = workflow.llm
    up = llm.reachable()
    return {
        "ollama": up,
        "story_model": llm.model,
        "story_model_installed": up and llm.installed(),
        "image_model": _image_id(),
        "vram_used_mb": vram_used_mb(),
        "queue": workflow.jobs.qsize(),
    }


class ModelChoice(BaseModel):
    story_model: str | None = Field(default=None, min_length=2, max_length=100, pattern=r"^[\w.\-:/]+$")
    image_model: str | None = None


@app.get("/api/models")
def models() -> dict:
    llm = workflow.llm
    installed = llm.models() if llm.reachable() else []
    names = {m["name"] for m in installed}
    return {
        "current": llm.model,
        "image": {"current": _image_id(), "options": _image_options()},
        "installed": installed,
        "suggested": [m for m in SUGGESTED if m not in names],  # downloaded on first use
    }


@app.put("/api/settings")
def set_model(body: ModelChoice) -> dict:
    """Pick the story and/or picture model for new work. Every logged call records the model that made it."""
    if any(s["state"] in ("queued", "running") for s in store.summaries()):
        raise HTTPException(409, "a story is being worked on; change the model when it is done")
    if body.image_model is not None:
        opt = next((o for o in _image_options() if o["id"] == body.image_model), None)
        if not opt:
            raise HTTPException(422, "unknown picture model")
        if not opt["usable"]:
            raise HTTPException(409, f"{opt['label']}: {opt['note']}")
        workflow.images = IMAGE_BACKENDS[body.image_model]
    if body.story_model:
        workflow.llm.use(body.story_model)
    PREFS.write_text(json.dumps({"story_model": workflow.llm.model, "image_model": _image_id()}), "utf-8")
    return health()


@app.delete("/api/stories/{sid}", status_code=204)
def delete(sid: str) -> Response:
    _idle(_get(sid))
    store.delete(sid)
    return Response(status_code=204)


@app.get("/api/stories/{sid}/book.pdf")
def book_pdf(sid: str, layout: Literal["pages", "booklet"] = "pages") -> Response:
    story = _get(sid)
    if not any(p.get("image") for p in story["pages"]):
        raise HTTPException(409, "no pictures yet")
    data = pdf.build(story, store.dir(sid), layout)
    name = f"{store.dir(sid).name[:40]}-{layout}.pdf"
    return Response(data, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"'})


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
