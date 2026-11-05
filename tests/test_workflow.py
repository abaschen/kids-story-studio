"""The whole workflow with fake models: no GPU, no Ollama."""

import io
import json
import re

from PIL import Image

from storystudio import prompts
from storystudio.config import Settings
from storystudio.store import Store
from storystudio.workflow import Workflow

STORY = "Title Line\n\n" + " ".join(f"word{i}" for i in range(400))


class FakeLLM:
    model = "fake:1b"

    def __init__(self, fail_first_review: bool = True, broken_drafts: int = 0, page_misses_moment: bool = False):
        self.reviews = 0
        self.fail_first_review = fail_first_review
        self.broken_drafts = broken_drafts
        self.page_misses_moment = page_misses_moment
        self.loaded = False
        self.calls: list[str] = []

    def info(self):
        return {"runtime": "fake", "model": self.model, "digest": "sha256:abc"}

    def installed(self):
        return True

    def reachable(self):
        return True

    def models(self):
        return [{"name": "fake:1b", "size_gb": 1.0}]

    def use(self, model):
        self.model = model

    def pull(self, on_progress):
        raise AssertionError("not needed")

    def unload(self):
        self.loaded = False

    def chat(self, messages, *, schema=None, temperature, seed):
        self.loaded = True
        meta = {"params": {"options": {"temperature": temperature, "seed": seed}}, "duration_s": 0.01}
        if schema is None:  # the writer: plain text
            self.calls.append("write")
            if self.broken_drafts:
                self.broken_drafts -= 1
                return "Title\n\nOnce upon a time 1. 2. 3. 4. 5. 6. 7. 8. 9. 10.", meta
            return STORY, meta
        if schema is prompts.BEATS_SCHEMA:
            self.calls.append("checklist")
            return json.dumps({"beats": ["find the rock", "ask the whale", " "]}), meta
        if schema is prompts.CRITIC_SCHEMA:
            self.reviews += 1
            ok = not (self.fail_first_review and self.reviews == 1)
            self.calls.append("critique")
            beat = {"evidence": "...", "note": ""}
            return json.dumps(
                {
                    "beats": [
                        beat | {"beat": "find the rock", "present": True},
                        beat | {"beat": "ask the whale", "present": ok},
                    ],
                    "problems": [] if ok else ["the whale is missing"],
                    "complete_arc": True,
                    "happy_or_hopeful_ending": True,
                    "teaching_moment": "sharing",
                    "age_appropriate": True,
                    "consistent": True,
                    "verdict": "pass" if ok else "revise",
                }
            ), meta
        if schema is prompts.BIBLE_SCHEMA:
            self.calls.append("bible")
            return json.dumps(
                {
                    "style": "soft watercolor",
                    "characters": [
                        {"id": "Mia", "name": "Mia", "role": "girl", "age": "2.5", "look": "red dress"},
                        {"id": "leo", "name": "Leo", "role": "boy", "age": "5", "look": "blue shirt"},
                    ],
                    "props": [{"id": "rock", "name": "rock", "look": "glowing"}],
                }
            ), meta
        if schema is prompts.PAGES_SCHEMA:
            self.calls.append("pages")
            fixing = "on no page yet" in messages[-1]["content"]
            n = 5 if fixing else 4
            pages = [
                {
                    "moments": [1] if (self.page_misses_moment and not fixing) else [1 + i % 2],
                    "text": f"Page {i}",
                    "illustration": f"scene {i}" + (" where Leo waves" if i == 3 else ""),
                    "characters": ["mia", "leo", "ghost", "mia"] if i != 3 else ["rock"],
                    "props": ["rock"],
                    "setting": "beach, day",
                }
                for i in range(n)
            ]
            return json.dumps({"pages": pages}), meta
        raise AssertionError(schema)


class FakeImages:
    def __init__(self, llm: FakeLLM):
        self.llm = llm
        self.loaded = False
        self.prompts: list[tuple[str, int]] = []

    def info(self):
        return {"runtime": "fake", "model": "fake/flux", "revision": "deadbeef"}

    def encode(self, prompts):
        assert not self.llm.loaded, "the story model must be unloaded before pictures"
        assert not self.loaded, "the encoder runs before the painter is loaded"
        self.encoded = set(prompts)
        return 0.1

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, prompt, *, refs, width, height, steps, guidance, seed):
        assert self.loaded and prompt in self.encoded
        self.prompts.append((prompt, len(refs)))
        return Image.new("RGB", (width // 16, height // 16), "orange"), {"params": {"seed": seed}, "duration_s": 0}


def make(tmp_path):
    llm = FakeLLM()
    images = FakeImages(llm)
    store = Store(tmp_path)
    return store, llm, images, Workflow(store, llm, images, Settings(stories_dir=tmp_path))


def test_full_book(tmp_path):
    store, llm, images, wf = make(tmp_path)
    s = store.create(context="Mia is 3 and Leo 6", prompt="Mia and Leo find a rock", language="English")
    wf.run_now(s["id"], "all")

    s = store.get(s["id"])
    # Checklist, draft, failed review, revision, passing review.
    assert llm.calls[:5] == ["checklist", "write", "critique", "write", "critique"]
    assert s["beats"] == ["find the rock", "ask the whale"]
    assert s["title"] == "Title Line" and s["story"].startswith("word0")
    assert [r["passed"] for r in s["reviews"]] == [False, True]
    assert s["input"]["prompt"] == "Mia and Leo find a rock"  # kept verbatim
    # Unknown character ids are dropped; ids are normalized.
    assert s["pages"][0]["characters"] == ["mia", "leo"]
    assert set(s["refs"]) == {"mia", "leo"}
    assert all(p["image"] == f"pages/{p['n']:02d}-v1.png" for p in s["pages"])
    assert all((store.dir(s["id"]) / p["image"]).exists() for p in s["pages"])
    # Pages get both character references; references themselves get none.
    assert [n for _, n in images.prompts] == [0, 0, 2, 2, 2, 1]
    assert not images.loaded  # freed after rendering

    calls = store.calls(s["id"])
    assert {c["kind"] for c in calls} == {"llm", "image", "image-encode"}
    assert all(c.get("digest") for c in calls if c["kind"] == "llm")
    assert all(c.get("revision") for c in calls if c["kind"] == "image")
    assert all("params" in c for c in calls)


def test_rerender_one_page_new_version(tmp_path):
    store, llm, images, wf = make(tmp_path)
    s = store.create(context="", prompt="A short test story idea", language="French")
    wf.run_now(s["id"], "all")
    images.prompts.clear()
    wf.run_now(s["id"], "render", pages=[2], new_seed=True)
    s = store.get(s["id"])
    assert s["pages"][1]["image"] == "pages/02-v2.png"
    assert s["pages"][0]["image"] == "pages/01-v1.png"
    assert len(images.prompts) == 1  # refs reused


def test_api_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("STORIES_DIR", str(tmp_path))
    import importlib

    from fastapi.testclient import TestClient

    import storystudio.config
    import storystudio.main

    importlib.reload(storystudio.config)
    main = importlib.reload(storystudio.main)
    llm = FakeLLM(fail_first_review=False)
    main.workflow.llm, main.workflow.images = llm, FakeImages(llm)
    with TestClient(main.app) as client:
        assert client.get("/").status_code == 200
        r = client.post("/api/stories", json={"context": "x", "prompt": "a story about a rock", "auto": True})
        sid = r.json()["id"]
        main.workflow.jobs.join()
        s = client.get(f"/api/stories/{sid}").json()
        assert s["status"]["state"] == "idle", s["status"]
        assert len(s["pages"]) == 4
        assert client.get(f"/files/{sid}/{s['pages'][0]['image']}").status_code == 200
        assert len(client.get(f"/api/stories/{sid}/calls").json()) > 5
        assert client.get("/api/stories/..%2Fetc").status_code == 404


def test_broken_draft_is_retried(tmp_path):
    store, llm, images, wf = make(tmp_path)
    llm.broken_drafts = 1
    s = store.create(context="", prompt="A short test story idea", language="English")
    wf.run_now(s["id"], "write")
    stages = [c["stage"] for c in store.calls(s["id"])]
    assert stages[:3] == ["checklist", "draft", "draft-retry1"]
    assert store.get(s["id"])["story"].startswith("word0")


def test_missing_moment_gets_a_page(tmp_path):
    store, llm, images, wf = make(tmp_path)
    llm.page_misses_moment = True
    s = store.create(context="", prompt="A short test story idea", language="English")
    wf.run_now(s["id"], "write")
    wf.run_now(s["id"], "plan")
    s = store.get(s["id"])
    assert "pages-fix-0" in [c["stage"] for c in store.calls(s["id"])]
    assert {m for p in s["pages"] for m in p["moments"]} == {1, 2}
    assert s["pages"][0]["characters"] == ["mia", "leo"]  # duplicates and unknown ids dropped


def test_named_characters_are_added(tmp_path):
    store, llm, images, wf = make(tmp_path)
    s = store.create(context="", prompt="A short test story idea", language="English")
    wf.run_now(s["id"], "write")
    wf.run_now(s["id"], "plan")
    assert store.get(s["id"])["pages"][3]["characters"] == ["leo"]  # named in the illustration only


def test_models_delete_and_pdf(tmp_path, monkeypatch):
    monkeypatch.setenv("STORIES_DIR", str(tmp_path))
    import importlib

    from fastapi.testclient import TestClient

    import storystudio.config
    import storystudio.main

    importlib.reload(storystudio.config)
    main = importlib.reload(storystudio.main)
    llm = FakeLLM(fail_first_review=False)
    main.workflow.llm, main.workflow.images = llm, FakeImages(llm)
    with TestClient(main.app) as client:
        assert client.get("/api/models").json()["current"] == "fake:1b"
        assert client.put("/api/settings", json={"story_model": "qwen3:14b"}).json()["story_model"] == "qwen3:14b"
        assert llm.model == "qwen3:14b"
        assert client.put("/api/settings", json={"story_model": "bad name!"}).status_code == 422
        new = {"context": "x", "prompt": "a story about a rock", "auto": True}
        sid = client.post("/api/stories", json=new).json()["id"]
        main.workflow.jobs.join()
        # cover + 4 pages + end = 6 A5 pages; the booklet pads to 8 and puts 4 on each of 2 sheets (4 sides).
        for layout, sides in (("pages", 6), ("booklet", 4)):
            r = client.get(f"/api/stories/{sid}/book.pdf?layout={layout}")
            assert r.status_code == 200 and r.content.startswith(b"%PDF")
            assert len(re.findall(rb"/Type\s*/Page\b(?!s)", r.content)) == sides
        assert client.delete(f"/api/stories/{sid}").status_code == 204
        assert client.get(f"/api/stories/{sid}").status_code == 404


def test_image_model_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("STORIES_DIR", str(tmp_path))
    monkeypatch.delenv("IDEOGRAM_API_KEY", raising=False)
    import importlib

    from fastapi.testclient import TestClient

    import storystudio.config
    import storystudio.main

    importlib.reload(storystudio.config)
    main = importlib.reload(storystudio.main)
    with TestClient(main.app) as client:
        img = client.get("/api/models").json()["image"]
        assert img["current"] == "flux-klein-4b"
        assert {o["id"]: o["usable"] for o in img["options"]} == {"flux-klein-4b": True, "ideogram-v3": False}
        assert client.put("/api/settings", json={"image_model": "ideogram-v3"}).status_code == 409  # no API key
        assert client.put("/api/settings", json={"image_model": "nope"}).status_code == 422
        ok = client.put("/api/settings", json={"image_model": "flux-klein-4b"}).json()
        assert ok["image_model"] == "flux-klein-4b"


def test_ideogram_request(monkeypatch):
    import httpx

    from storystudio.images import IdeogramV3

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.ideogram.ai":
            seen["key"] = request.headers["api-key"]
            seen["body"] = request.content
            return httpx.Response(200, json={"data": [{"url": "https://cdn.test/a.png", "resolution": "1024x768"}]})
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buf, "PNG")
        return httpx.Response(200, content=buf.getvalue())

    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    painter = IdeogramV3("secret")
    ref = Image.new("RGB", (8, 8))
    image, meta = painter.generate("a whale", refs=[ref, ref], width=1024, height=768, steps=4, guidance=1, seed=5)
    assert image.size == (8, 8) and seen["key"] == "secret"
    assert b'name="resolution"\r\n\r\n1024x768' in seen["body"] and b"character_reference_images" in seen["body"]
    assert meta["params"]["reference_images_sent"] == 1 and meta["params"]["reference_images_available"] == 2
    assert "secret" not in json.dumps(meta) + json.dumps(painter.info())
