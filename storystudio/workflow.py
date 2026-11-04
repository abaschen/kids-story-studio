"""The book workflow, one GPU job at a time:

    write  -> draft, then critique/revise until the editor passes it (or max rounds)
    revise -> the parent's own feedback, then the same critique loop
    plan   -> visual bible (style, characters, props) + pages (text + illustration), then a continuity check
    render -> unload the story model, load FLUX.2 [klein]: one reference per character, then every page
    all    -> write + plan + render

Every model call is appended to calls.jsonl with its inputs, parameters, model version and output.
"""

import json
import logging
import queue
import re
import threading
import traceback
from typing import Any, Protocol

from PIL import Image

from . import prompts
from .config import Settings
from .store import Store

log = logging.getLogger("storystudio")


class StoryModel(Protocol):
    model: str

    def info(self) -> dict[str, Any]: ...
    def installed(self) -> bool: ...
    def pull(self, on_progress) -> None: ...
    def chat(self, messages, *, schema=None, temperature: float, seed: int) -> tuple[str, dict[str, Any]]: ...
    def unload(self) -> None: ...


class ImageModel(Protocol):
    def info(self) -> dict[str, Any]: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, prompt: str, *, refs, width, height, steps, guidance, seed) -> tuple[Image.Image, dict]: ...


def vram() -> int | None:
    from .images import vram_used_mb

    return vram_used_mb()


class Workflow:
    def __init__(self, store: Store, llm: StoryModel, images: ImageModel, cfg: Settings):
        self.store, self.llm, self.images, self.cfg = store, llm, images, cfg
        self.jobs: queue.Queue[tuple[str, str, dict]] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="workflow")

    def start(self) -> None:
        self.thread.start()

    def submit(self, sid: str, kind: str, **args: Any) -> None:
        self.store.status(sid, state="queued", step=f"waiting to {kind}", error="", done=0, total=0)
        self.jobs.put((sid, kind, args))

    # -- worker --------------------------------------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            sid, kind, args = self.jobs.get()
            try:
                self.store.status(sid, state="running")
                getattr(self, f"_job_{kind}")(sid, **args)
                self.store.status(sid, state="idle", step="done", error="")
            except Exception as e:  # keep the worker alive, show the error in the UI
                log.exception("job %s %s failed", kind, sid)
                self.store.status(sid, state="error", error=f"{type(e).__name__}: {e}", step=traceback.format_exc(-1))
            finally:
                self.jobs.task_done()

    def run_now(self, sid: str, kind: str, **args: Any) -> None:
        """Synchronous variant (tests, CLI)."""
        getattr(self, f"_job_{kind}")(sid, **args)

    def _job_all(self, sid: str) -> None:
        self._job_write(sid)
        self._job_plan(sid)
        self._job_render(sid)

    # -- story model ---------------------------------------------------------------------------------------------

    def _ensure_llm(self, sid: str) -> None:
        if self.llm.installed():
            return

        def progress(status: str, done: int, total: int) -> None:
            mb = lambda b: round(b / 1e6)  # noqa: E731
            self.store.status(sid, step=f"downloading {self.llm.model}: {status}", done=mb(done), total=mb(total))

        self.store.status(sid, step=f"downloading {self.llm.model} (first run)")
        self.llm.pull(progress)

    def _chat(
        self,
        sid: str,
        stage: str,
        system: str,
        user: str | list[dict],
        *,
        schema: dict | None,
        temperature: float,
        seed_offset: int,
    ) -> Any:
        """One call to the story model, logged. JSON in, parsed JSON out; with no schema, plain text."""
        story = self.store.get(sid)
        messages = [{"role": "system", "content": system}]
        messages += [{"role": "user", "content": user}] if isinstance(user, str) else user
        seed = story["seed"] + seed_offset
        text, meta = self.llm.chat(messages, schema=schema, temperature=temperature, seed=seed)
        out, err = text, None
        if schema:
            try:
                out = json.loads(text)
            except json.JSONDecodeError as e:
                out, err = None, str(e)
        self.store.log_call(
            sid,
            {
                "kind": "llm",
                "stage": stage,
                **self.llm.info(),
                **meta,
                "vram_used_mb": vram(),
                "request": {"messages": messages, "schema": schema},
                "response": text,
                "parse_error": err,
            },
        )
        if out is None:
            raise ValueError(f"{stage}: the story model did not return valid JSON ({err})")
        return out

    def _write(self, sid: str, stage: str, messages: list[dict], seed_offset: int) -> None:
        """Writer call (plain text). A derailed answer is retried with another seed, at most twice."""
        story = self.store.get(sid)
        writer = prompts.WRITER_SYSTEM.format(language=story["input"]["language"])
        broken = None
        for attempt in range(3):
            text = self._chat(
                sid,
                stage if not attempt else f"{stage}-retry{attempt}",
                writer,
                messages,
                schema=None,
                temperature=self.cfg.temperature,
                seed_offset=seed_offset + 50 * attempt,
            )
            title, body = prompts.parse_story(text)
            broken = prompts.looks_broken(body)
            if not broken:
                self.store.update(sid, title=title, story=body)
                return
            log.warning("%s %s: draft %s, retrying", sid, stage, broken)
            self.store.status(sid, step=f"the draft came out {broken}, trying again")
        raise ValueError(f"{stage}: the story model kept producing a broken story ({broken})")

    def _writer_messages(self, story: dict) -> list[dict]:
        req = prompts.user_request(story["input"]["context"], story["input"]["prompt"])
        checklist = prompts.checklist(story["beats"])
        return [{"role": "user", "content": f"{req}\n\nChecklist (every moment, in order):\n{checklist}"}]

    def _draft_message(self, story: dict) -> dict:
        return {"role": "assistant", "content": f"{story['title']}\n\n{story['story']}"}

    def _refine(self, sid: str) -> None:
        """Editor review, then revise, ... always ending on a review. Stops early when a review passes."""
        rounds = self.cfg.max_refine_rounds
        for i in range(rounds):
            story = self.store.get(sid)
            r = len(story["reviews"])
            self.store.status(sid, step=f"editor review, round {i + 1}", done=i + 1, total=rounds)
            req = prompts.user_request(story["input"]["context"], story["input"]["prompt"])
            review = self._chat(
                sid,
                f"critique-{r}",
                prompts.CRITIC_SYSTEM,
                f"{req}\n\nChecklist:\n{prompts.checklist(story['beats'])}\n\n"
                f"Draft title: {story['title']}\n\nDraft story:\n{story['story']}",
                schema=prompts.CRITIC_SCHEMA,
                temperature=self.cfg.critic_temperature,
                seed_offset=100 + r,
            )
            review["problems"] = [p for p in review["problems"] if not _empty(p)]
            missing = [b["beat"] for b in review["beats"] if not b["present"]]
            checks = ("complete_arc", "happy_or_hopeful_ending", "age_appropriate", "consistent")
            failed = [k for k in checks if not review[k]]
            passed = review["verdict"] == "pass" and not missing and not failed and not review["problems"]
            review |= {"round": r, "passed": passed}
            self.store.update(sid, reviews=story["reviews"] + [review])
            if passed or i == rounds - 1:
                return
            problems = review["problems"] + [f"Missing or incomplete: {b}" for b in missing]
            problems += [f"Check failed: {k.replace('_', ' ')}" for k in failed]
            self.store.status(sid, step=f"revising, round {i + 1}")
            fix = prompts.REVISE_INSTRUCTION.format(problems="\n".join(f"- {p}" for p in dict.fromkeys(problems)))
            messages = self._writer_messages(story) + [self._draft_message(story), {"role": "user", "content": fix}]
            self._write(sid, f"revise-{r}", messages, seed_offset=200 + r)

    def _job_write(self, sid: str) -> None:
        self._ensure_llm(sid)
        story = self.store.get(sid)
        self.store.status(sid, step="listing the moments you asked for", done=0, total=0)
        beats = self._chat(
            sid,
            "checklist",
            prompts.BEATS_SYSTEM,
            prompts.user_request(story["input"]["context"], story["input"]["prompt"]),
            schema=prompts.BEATS_SCHEMA,
            temperature=self.cfg.critic_temperature,
            seed_offset=10,
        )["beats"]
        beats = [re.sub(r"^\s*\d+[.)]\s*", "", b).strip() for b in beats]
        beats = [b for b in beats if b]
        story = self.store.update(sid, beats=beats, reviews=[], bible=None, pages=[], refs={})
        self.store.status(sid, step="writing the first draft")
        self._write(sid, "draft", self._writer_messages(story), seed_offset=0)
        self._refine(sid)

    def _job_revise(self, sid: str, feedback: str) -> None:
        """The parent's notes, then the editor loop again."""
        self._ensure_llm(sid)
        story = self.store.get(sid)
        if not story.get("beats"):
            raise ValueError("write the story first")
        self.store.status(sid, step="revising with your notes")
        fix = prompts.REVISE_INSTRUCTION.format(problems=f"- {feedback}")
        messages = self._writer_messages(story) + [self._draft_message(story), {"role": "user", "content": fix}]
        r = len(story["reviews"])
        self._write(sid, f"parent-revise-{r}", messages, seed_offset=300 + r)
        self.store.update(sid, pages=[], bible=None, refs={}, feedback=story.get("feedback", []) + [feedback])
        self._refine(sid)

    def _job_plan(self, sid: str) -> None:
        self._ensure_llm(sid)
        story = self.store.get(sid)
        lang = story["input"]["language"]
        beats = story.get("beats") or []
        base = f"Family context:\n{story['input']['context']}\n\nTitle: {story['title']}\n\nStory:\n{story['story']}"

        self.store.status(sid, step="art direction: style and characters", done=1, total=3)
        bible = self._chat(
            sid, "bible", prompts.BIBLE_SYSTEM, base, schema=prompts.BIBLE_SCHEMA, temperature=0.5, seed_offset=400
        )
        for c in bible["characters"]:
            c["id"] = _clean_id(c["id"] or c["name"])
        for p in bible["props"]:
            p["id"] = _clean_id(p["id"] or p["name"])

        self.store.status(sid, step="splitting into pages", done=2, total=3)
        cast = json.dumps({"characters": bible["characters"], "props": bible["props"]}, ensure_ascii=False)
        user = f"{base}\n\nVisual bible (use these ids):\n{cast}\n\nChecklist:\n{prompts.checklist(beats)}"
        system = prompts.PAGES_SYSTEM.format(language=lang)
        pages = self._chat(sid, "pages", system, user, schema=prompts.PAGES_SCHEMA, temperature=0.4, seed_offset=500)
        pages = pages["pages"]

        self.store.status(sid, step="checking the pages tell the whole story", done=3, total=3)
        checked = self._chat(
            sid,
            "pages-check",
            prompts.PAGES_CHECK_SYSTEM,
            f"{user}\n\nPages:\n{json.dumps({'pages': pages}, ensure_ascii=False)}",
            schema=prompts.PAGES_SCHEMA,
            temperature=0.2,
            seed_offset=600,
        )["pages"]
        if len(checked) >= max(3, len(pages) // 2):  # guard against a truncated answer
            pages = checked

        # Every checklist moment must be on some page; ask for the missing ones (twice at most).
        for attempt in range(2):
            covered = {m for p in pages for m in p.get("moments", [])}
            missing = [f"{i}. {b}" for i, b in enumerate(beats, 1) if i not in covered]
            if not missing:
                break
            self.store.status(sid, step=f"adding pages for {len(missing)} missing moment(s)")
            fixed = self._chat(
                sid,
                f"pages-fix-{attempt}",
                system,
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": json.dumps({"pages": pages}, ensure_ascii=False)},
                    {"role": "user", "content": prompts.PAGES_FIX.format(missing="\n".join(missing))},
                ],
                schema=prompts.PAGES_SCHEMA,
                temperature=0.2,
                seed_offset=700 + attempt,
            )["pages"]
            if len(fixed) >= len(pages):
                pages = fixed

        prop_ids = {p["id"] for p in bible["props"]}
        clean = [
            {
                "n": i + 1,
                "text": p["text"].strip(),
                "illustration": p["illustration"].strip(),
                "characters": _cast(p, bible["characters"]),
                "props": list(dict.fromkeys(x for x in map(_clean_id, p["props"]) if x in prop_ids)),
                "setting": p["setting"].strip(),
                "moments": [m for m in p.get("moments", []) if 1 <= m <= len(beats)],
                "image": None,
            }
            for i, p in enumerate(pages)
        ]
        self.store.update(sid, bible=bible, pages=clean, refs={})

    # -- pictures ------------------------------------------------------------------------------------------------

    def _image(
        self, sid: str, stage: str, prompt: str, refs: list[tuple[str, str]], size: tuple[int, int], seed: int, out: str
    ) -> None:
        d = self.store.dir(sid)
        ref_imgs = [Image.open(d / path).convert("RGB") for _, path in refs]
        img, meta = self.images.generate(
            prompt,
            refs=ref_imgs,
            width=size[0],
            height=size[1],
            steps=self.cfg.image_steps,
            guidance=self.cfg.image_guidance,
            seed=seed,
        )
        img.save(d / out)
        self.store.log_call(
            sid,
            {
                "kind": "image",
                "stage": stage,
                **self.images.info(),
                **meta,
                "vram_used_mb": vram(),
                "request": {"prompt": prompt, "reference_images": [{"id": i, "path": p} for i, p in refs]},
                "output": out,
            },
        )

    def _job_render(self, sid: str, pages: list[int] | None = None, new_seed: bool = False) -> None:
        story = self.store.get(sid)
        if not story["pages"]:
            raise ValueError("no pages yet: plan the book first")
        bible = story["bible"]
        chars = {c["id"]: c for c in bible["characters"]}
        props = {p["id"]: p for p in bible["props"]}
        todo = [p for p in story["pages"] if pages is None or p["n"] in pages]

        # Every picture of this run, worked out up front so all prompts are encoded in one pass.
        # (stage, prompt, reference ids, size, seed, output path, page number or None)
        jobs: list[tuple[str, str, list[str], tuple[int, int], int, str, int | None]] = []
        for cid in sorted({c for p in todo for c in p["characters"]} - set(story["refs"])):
            seed = story["seed"] + 1000 + list(chars).index(cid)
            prompt = prompts.ref_prompt(bible["style"], chars[cid])
            jobs.append((f"ref-{cid}", prompt, [], self.cfg.ref_size, seed, f"refs/{cid}.png", None))
        for page in todo:
            n = page["n"]
            present = [chars[c] for c in page["characters"]]  # most important first
            prompt = prompts.image_prompt(
                bible["style"], page["illustration"], page["setting"], present, [props[x] for x in page["props"]]
            )
            version = (page.get("version") or 0) + 1
            seed = story["seed"] + 2000 + n + (version * 7919 if new_seed else 0)
            ref_ids = [c["id"] for c in present[: self.cfg.max_refs_per_page]]
            jobs.append((f"page-{n}", prompt, ref_ids, self.cfg.page_size, seed, f"pages/{n:02d}-v{version}.png", n))

        try:
            self.store.status(sid, step="freeing the GPU (unloading the story model)", done=0, total=len(jobs))
            self.llm.unload()
            self.store.status(sid, step=f"reading {len(jobs)} picture descriptions (text encoder)")
            took = self.images.encode([j[1] for j in jobs])
            self.store.log_call(
                sid,
                {
                    "kind": "image-encode",
                    "stage": "encode-prompts",
                    **self.images.info(),
                    "duration_s": took,
                    "params": {"prompts": len(jobs)},
                    "vram_used_mb": vram(),
                },
            )
            self.store.status(sid, step="loading the painter (image transformer)")
            self.images.load()

            refs = dict(story["refs"])
            for i, (stage, prompt, ref_ids, size, seed, out, n) in enumerate(jobs):
                what = f"painting page {n} of {len(story['pages'])}" if n else f"drawing {chars[stage[4:]]['name']}"
                self.store.status(sid, step=what, done=i, total=len(jobs))
                self._image(sid, stage, prompt, [(r, refs[r]) for r in ref_ids], size, seed, out)
                s = self.store.get(sid)
                if n is None:
                    refs[stage[4:]] = out
                    self.store.update(sid, refs=refs)
                    continue
                for p in s["pages"]:
                    if p["n"] == n:
                        p["image"], p["version"] = out, int(out.rsplit("-v", 1)[1].split(".")[0])
                self.store.update(sid, pages=s["pages"])
            self.store.status(sid, done=len(jobs), total=len(jobs))
        finally:
            self.store.status(sid, step="freeing the GPU (unloading the picture model)")
            self.images.unload()


def _clean_id(s: str) -> str:
    return "".join(ch for ch in s.lower().strip() if ch.isalnum() or ch in "-_") or "x"


def _empty(problem: str) -> bool:
    """Models write "None" or "N/A" for an empty list item."""
    return problem.strip().strip(".").lower() in ("", "none", "n/a", "na", "no problems", "nothing")


def _cast(page: dict, characters: list[dict]) -> list[str]:
    """Characters of a page: the listed ones (known ids, in order), then anyone the illustration names."""
    ids = {c["id"] for c in characters}
    listed = [c for c in map(_clean_id, page["characters"]) if c in ids]
    text = page["illustration"].lower()
    named = [c["id"] for c in characters if re.search(rf"\b{re.escape(c['name'].lower())}\b", text)]
    return list(dict.fromkeys(listed + named))
