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
