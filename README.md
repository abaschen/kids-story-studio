# Kids Story Studio

Turn a short story idea into an illustrated picture book for young children. Everything runs on your own PC with one
NVIDIA GPU: no cloud, no account, no API key. Built and tested on an RTX 4080 (16 GB) with 32 GB of RAM.

1. **Write.** You give some context (who the children are, their ages) and the story idea. The story model writes a
   draft, then an editor pass checks it against your request: every event you asked for is there and in order, the
   ending is happy or hopeful or teaches something, and it fits the youngest child's age. It revises until the check
   passes (3 rounds at most). You can edit the text or ask for changes ("make the turtle funnier").
2. **Plan.** The story becomes 10 to 16 pages. Each page has the text read aloud and an illustration that shows what
   the text doesn't say. A visual bible keeps the art style and every character's look fixed, and a last pass checks
   the pages still tell the whole story in order.
3. **Paint.** The story model is unloaded, the picture model is loaded. It first draws one reference picture per
   character, then every page with those references, so the children look the same from page to page.

Every step is saved in its own folder under `stories/`, with a full log of each model call.

## Models

| Step | Model | Runs in | Size |
|---|---|---|---|
| Write and plan | [Gemma 4 12B](https://ollama.com/library/gemma4) (`gemma4:12b`, Q4) | Ollama | 8 GB |
| Pictures | [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (Apache 2.0), 4 steps, multi-reference | diffusers | 16 GB download; runs in two passes (text encoder, then image transformer), ~8 GB each |

Only one model is on the GPU at a time. Both are set in `compose.yaml` and can be changed with environment variables
(`STORY_MODEL`, `IMAGE_MODEL`, `IMAGE_MODEL_REVISION`); for example `STORY_MODEL=mistral-small3.2:24b` writes better
prose but spills out of 16 GB of VRAM and gets slower. The picture model is pinned to a Hugging Face commit so a
"model version" in the log means one exact set of weights.

Why not in the browser? The story model could run there (WebLLM), but no in-browser image model comes close to FLUX.2
klein, saving a book with its log needs a local process anyway, and two runtimes sharing one 16 GB card can't take
turns cleanly.

## Run it

Needs Docker with GPU support: Docker Desktop on Windows (WSL 2 backend) or Docker Engine with the NVIDIA Container
Toolkit on Linux.

```bash
docker compose up -d --build
```

Open <http://localhost:8765>. The first story downloads the models (about 25 GB in total, into Docker volumes; on
Windows they live in Docker Desktop's disk image, usually on C:). With [just](https://github.com/casey/just):
`just up`, `just logs`, `just down`.

The server listens on `127.0.0.1` only. Change the port with `PORT=9000 docker compose up -d`.

## Using it

- **New story**: context (saved for next time), the idea, the language of the text. Tick *Make the whole book without
  stopping* to go straight from idea to finished book.
- **1 · Story**: the text and each editor review (every beat checked ✓/✗, problems found, the lesson). Edit by hand,
  ask for a revision, or approve to make pages.
- **2 · Pages**: the art style, each character with their reference picture, and every page (text, illustration
  description, picture). Edit text or illustrations, paint every page, or paint one page again with a new seed.
- **3 · Book**: read it page by page (arrow keys work).
- **Share**: the Book tab has a PDF to read on screen and a print-ready PDF booklet (A4 landscape, print
  double-sided flipping on the short edge, fold in half, staple). 🗑 Delete removes a story and its files.
- **Story model**: the picker in the sidebar lists the models Ollama has; others download on first use. It applies to
  the next job and is remembered. Every log line records the model that made it.
- **Log**: every call with model, version, parameters, duration, and the full request and response.

Example idea:

> Context: Mia is 3 and her brother Leo is 6.
>
> Mia and Leo find a tiny lost cloud in the garden. It's too small to rain and it's sad. They try to help it grow:
> they ask the watering can, the pond, then the river, and each one sends them further, until the sea teaches the
> cloud how to drink the sunshine. The little cloud grows and waters Grandma's garden, and the children learn that
> everyone grows at their own pace.

## What gets saved

```
stories/20261104-153012-mia-and-leo-find-a-tiny-lost-cloud-1a2b/
  story.json    the request (verbatim), seed, story text, reviews, visual bible, pages, current state
  calls.jsonl   one line per model call or hand edit
  refs/         one reference picture per character
  pages/        01-v1.png, 02-v1.png, ... (painting a page again adds -v2, nothing is overwritten)
```

Each line of `calls.jsonl` has the step, the time, `runtime` (Ollama or diffusers/torch versions), `model` plus its
Ollama `digest` or Hugging Face `revision`, all parameters (temperature, seed, context size; or size, steps,
guidance, seed, number of reference pictures), the duration, GPU memory in use, and the full request and output.

`stories/` is in `.gitignore`: stories about your own children stay on your disk.

## Develop

The UI is plain HTML/JS in `storystudio/static/`; the server is FastAPI. Tests run the whole workflow with fake
models, no GPU needed:

```bash
uv sync
uv run pytest -q
```

To run the server outside Docker, start Ollama, then `uv sync --extra gpu` and
`uv run uvicorn storystudio.main:app --port 8765` (the GPU extra pulls the CUDA 13 build of PyTorch).

## License

MIT for this code. The models come with their own licenses: FLUX.2 [klein] 4B is Apache 2.0; check the Gemma 4 model
page for its terms.
