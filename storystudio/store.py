"""One folder per story: story.json (state), calls.jsonl (every model call), refs/ and pages/ (images)."""

import json
import re
import shutil
import threading
import time
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_lock = threading.RLock()


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def slugify(text: str, n: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:n].strip("-") or "story"


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, sid: str) -> Path:
        if not re.fullmatch(r"[a-z0-9-]+", sid):
            raise KeyError(sid)
        return self.root / sid

    def create(self, *, context: str, prompt: str, language: str, seed: int | None = None) -> dict[str, Any]:
        sid = f"{time.strftime('%Y%m%d-%H%M%S')}-{slugify(prompt)}-{uuid.uuid4().hex[:4]}"
        story = {
            "id": sid,
            "created": now(),
            "updated": now(),
            # The parent's input, kept verbatim.
            "input": {"context": context, "prompt": prompt, "language": language},
            "seed": seed if seed is not None else int.from_bytes(uuid.uuid4().bytes[:4], "big") % 2**31,
            "title": "",
            "story": "",
            "beats": [],  # the moments the parent asked for, as a checklist
            "reviews": [],  # one editor verdict per round
            "bible": None,  # art style + characters
            "pages": [],  # [{n, text, illustration, characters, setting, image}]
            "refs": {},  # character id -> image path
            "status": {"state": "idle", "step": "", "done": 0, "total": 0, "error": ""},
        }
        (self.root / sid / "pages").mkdir(parents=True)
        (self.root / sid / "refs").mkdir()
        self.save(story)
        return story

    def get(self, sid: str) -> dict[str, Any]:
        path = self.dir(sid) / "story.json"
        if not path.exists():
            raise KeyError(sid)
        with _lock:
            return json.loads(path.read_text("utf-8"))

    def save(self, story: dict[str, Any]) -> None:
        story["updated"] = now()
        path = self.dir(story["id"]) / "story.json"
        tmp = path.with_suffix(".tmp")
        with _lock:
            tmp.write_text(json.dumps(story, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(path)

    def update(self, sid: str, **changes: Any) -> dict[str, Any]:
        with _lock:
            story = self.get(sid)
            story.update(changes)
            self.save(story)
            return story

    def status(self, sid: str, **changes: Any) -> None:
        with _lock:
            story = self.get(sid)
            story["status"].update(changes)
            self.save(story)

    def delete(self, sid: str) -> None:
        d = self.dir(sid)
        if not (d / "story.json").exists():
            raise KeyError(sid)
        shutil.rmtree(d)

    def summaries(self) -> list[dict[str, Any]]:
        out = []
        for p in sorted(self.root.glob("*/story.json"), reverse=True):
            try:
                s = json.loads(p.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            cover = next((pg["image"] for pg in s.get("pages", []) if pg.get("image")), None)
            out.append(
                {
                    "id": s["id"],
                    "title": s.get("title") or s["input"]["prompt"][:60],
                    "created": s["created"],
                    "pages": len(s.get("pages", [])),
                    "cover": cover,
                    "state": s["status"]["state"],
                }
            )
        return out

    def log_call(self, sid: str, record: dict[str, Any]) -> None:
        """Append one model call (inputs, parameters, model version, output) to calls.jsonl."""
        record = {"id": uuid.uuid4().hex[:12], "at": now(), **record}
        with _lock, open(self.dir(sid) / "calls.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def calls(self, sid: str) -> list[dict[str, Any]]:
        path = self.dir(sid) / "calls.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
