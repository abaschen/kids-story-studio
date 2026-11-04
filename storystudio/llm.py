"""Story model through Ollama. Every reply comes back with what's needed to log it: model digest and options."""

import json
import time
from collections.abc import Callable
from typing import Any

import httpx


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str, model: str, num_ctx: int):
        self.url = url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self._info: dict[str, Any] | None = None

    def _client(self, timeout: float = 900) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=httpx.Timeout(timeout, connect=10))

    def reachable(self) -> bool:
        try:
            with self._client(5) as c:
                return c.get("/api/version").status_code == 200
        except httpx.HTTPError:
            return False

    def version(self) -> str:
        with self._client(10) as c:
            return c.get("/api/version").json().get("version", "?")

    def installed(self) -> bool:
        with self._client(10) as c:
            names = {m["name"] for m in c.get("/api/tags").json().get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def pull(self, on_progress: Callable[[str, int, int], None]) -> None:
        """Download the model on first use, reporting bytes done / total."""
        with self._client(None) as c, c.stream("POST", "/api/pull", json={"model": self.model}) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                if "error" in msg:
                    raise LLMError(msg["error"])
                on_progress(msg.get("status", ""), msg.get("completed", 0), msg.get("total", 0))
        self._info = None

    def info(self) -> dict[str, Any]:
        """Exact model version for the log: the Ollama digest plus quantization details."""
        if self._info is None:
            with self._client(30) as c:
                tags = c.get("/api/tags").json().get("models", [])
                tag = next((m for m in tags if m["name"] in (self.model, f"{self.model}:latest")), {})
                self._info = {
                    "runtime": f"ollama {self.version()}",
                    "model": self.model,
                    "digest": tag.get("digest"),
                    "details": tag.get("details", {}),
                }
        return self._info

    def chat(
        self, messages: list[dict[str, str]], *, schema: dict[str, Any] | None = None, temperature: float, seed: int
    ) -> tuple[str, dict[str, Any]]:
        options = {"temperature": temperature, "num_ctx": self.num_ctx, "seed": seed}
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        body["think"] = False  # straight answers; the critique step does the reasoning
        if schema:
            body["format"] = schema
        t0 = time.perf_counter()
        with self._client() as c:
            r = c.post("/api/chat", json=body)
            if r.status_code == 400 and "think" in r.text:  # model without a thinking switch
                body.pop("think")
                r = c.post("/api/chat", json=body)
        if r.status_code != 200:
            raise LLMError(f"ollama {r.status_code}: {r.text[:300]}")
        data = r.json()
        meta = {
            "params": {k: body[k] for k in ("options", "think") if k in body}
            | {"format": "json_schema" if schema else None},
            "duration_s": round(time.perf_counter() - t0, 2),
            "tokens_in": data.get("prompt_eval_count"),
            "tokens_out": data.get("eval_count"),
        }
        return data["message"]["content"], meta

    def unload(self) -> None:
        """Free the GPU before pictures: keep_alive 0 evicts the model now."""
        try:
            with self._client(60) as c:
                c.post("/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass
