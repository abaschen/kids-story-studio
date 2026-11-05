"""Pictures with FLUX.2 [klein] 4B through diffusers, in two passes so it fits a 16 GB card *and* a 16 GB Docker VM:

    encode(prompts)  text encoder alone on the GPU: every prompt of the book -> embeddings (kept on the CPU), freed
    generate(...)    transformer + VAE alone on the GPU, painting from those embeddings

Each half is ~8 GB, so neither the GPU nor RAM ever holds the whole model, and nothing moves between them per image.
"""

import gc
import shutil
import subprocess
import time
from typing import Any

from PIL import Image


def vram_used_mb() -> int | None:
    """GPU memory in use on the whole card (both containers), for the log."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(out.stdout.split()[0])
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None


def _free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class FluxKlein:
    max_sequence_length = 512

    def __init__(self, repo: str, revision: str):
        self.repo = repo
        self.revision = revision
        self.pipe = None  # transformer + VAE, no text encoder
        self.embeds: dict[str, Any] = {}
        self._versions: dict[str, Any] | None = None

    def info(self) -> dict[str, Any]:
        if self._versions is None:
            import diffusers
            import torch

            self._versions = {
                "runtime": f"diffusers {diffusers.__version__}, torch {torch.__version__}",
                "model": self.repo,
                "revision": self.revision,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            }
        return self._versions

    def _pipeline(self, **skip: None):
        import torch
        from diffusers import Flux2KleinPipeline

        return Flux2KleinPipeline.from_pretrained(
            self.repo, revision=self.revision, torch_dtype=torch.bfloat16, **skip
        ).to("cuda")

    def encode(self, prompts: list[str]) -> float:
        """Text encoder pass. Returns the time it took."""
        import torch

        todo = [p for p in dict.fromkeys(prompts) if p not in self.embeds]
        if not todo:
            return 0.0
        t0 = time.perf_counter()
        enc = self._pipeline(transformer=None, vae=None)
        with torch.inference_mode():
            for p in todo:
                emb, _ = enc.encode_prompt(prompt=p, device="cuda", max_sequence_length=self.max_sequence_length)
                self.embeds[p] = emb.to("cpu")
        del enc
        _free()
        return round(time.perf_counter() - t0, 2)

    def load(self) -> None:
        if self.pipe is None:
            self.pipe = self._pipeline(text_encoder=None, tokenizer=None)

    def unload(self) -> None:
        self.pipe = None
        self.embeds.clear()
        _free()

    def generate(
        self, prompt: str, *, refs: list[Image.Image], width: int, height: int, steps: int, guidance: float, seed: int
    ) -> tuple[Image.Image, dict[str, Any]]:
        import torch

        if prompt not in self.embeds:
            raise RuntimeError("prompt was not encoded first (call encode() before load())")
        self.load()
        t0 = time.perf_counter()
        with torch.inference_mode():
            image = self.pipe(
                prompt_embeds=self.embeds[prompt].to("cuda"),
                image=refs or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
        meta = {
            "params": {
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance,
                "seed": seed,
                "dtype": "bfloat16",
                "max_sequence_length": self.max_sequence_length,
                "reference_images": len(refs),
            },
            "duration_s": round(time.perf_counter() - t0, 2),
        }
        return image, meta


class IdeogramV3:
    """Ideogram 3.0 over its web API. Unlike the local model, every picture description (and the character picture
    used as reference) leaves this PC. It accepts a single character reference per picture, so the most important
    character of a page is passed as the reference and the others are described in the prompt only."""

    url = "https://api.ideogram.ai/v1/ideogram-v3/generate"

    def __init__(self, api_key: str, speed: str = "DEFAULT"):
        self.api_key = api_key
        self.speed = speed

    def info(self) -> dict[str, Any]:
        return {
            "runtime": "ideogram web api v1 (ideogram-v3/generate)",
            "model": "ideogram-v3",
            "revision": f"api, rendering_speed={self.speed}",
            "gpu": None,
        }

    def encode(self, prompts: list[str]) -> float:
        return 0.0

    def load(self) -> None:
        pass

    def unload(self) -> None:
        pass

    def generate(
        self, prompt: str, *, refs: list[Image.Image], width: int, height: int, steps: int, guidance: float, seed: int
    ) -> tuple[Image.Image, dict[str, Any]]:
        import io
        import time

        import httpx

        if not self.api_key:
            raise RuntimeError("IDEOGRAM_API_KEY is not set (add it to .env and restart the app)")
        data = {
            "prompt": prompt,
            "resolution": f"{width}x{height}",
            "seed": str(seed % 2_147_483_647),
            "rendering_speed": self.speed,
            "magic_prompt": "OFF",  # paint what the story says, don't let the API rewrite it
            "num_images": "1",
        }
        files = []
        if refs:
            buf = io.BytesIO()
            refs[0].save(buf, "PNG")
            files.append(("character_reference_images", ("character.png", buf.getvalue(), "image/png")))
        t0 = time.perf_counter()
        with httpx.Client(timeout=180) as c:
            for attempt in range(4):
                r = c.post(self.url, headers={"Api-Key": self.api_key}, data=data, files=files or None)
                if r.status_code != 429:
                    break
                time.sleep(5 * 2**attempt)
            if r.status_code != 200:
                raise RuntimeError(f"Ideogram error {r.status_code}: {r.text[:300]}")
            item = r.json()["data"][0]
            if not item.get("url"):
                raise RuntimeError("Ideogram refused this picture (safety check); edit the illustration and retry")
            image = Image.open(io.BytesIO(c.get(item["url"]).content)).convert("RGB")
        meta = {
            "params": {
                "resolution": data["resolution"],
                "seed": int(data["seed"]),
                "rendering_speed": self.speed,
                "magic_prompt": "OFF",
                "reference_images_sent": len(files),
                "reference_images_available": len(refs),
                "returned_resolution": item.get("resolution"),
                "style_type": item.get("style_type"),
            },
            "duration_s": round(time.perf_counter() - t0, 2),
        }
        return image, meta


FluxKlein.id, FluxKlein.label, FluxKlein.cloud = "flux-klein-4b", "FLUX.2 klein 4B (local)", False
IdeogramV3.id, IdeogramV3.label, IdeogramV3.cloud = "ideogram-v3", "Ideogram 3.0 (cloud, API key)", True
