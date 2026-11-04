"""Prompts and JSON schemas for each story step. Illustration prompts are always English; story text follows the
chosen language. Prose is written as plain text (long prose inside constrained JSON tends to derail); everything the
code needs to read back is JSON with a schema."""

import re

WRITER_SYSTEM = """You are a warm, award-winning author of picture books for young children.

Write the story the parent describes. Rules:
- Keep EVERY moment in the checklist, in the same order, with every detail the parent gave (counts, which animal,
  how something happens). You may add small details, gentle dialogue, sounds and feelings, but never drop, merge or
  soften a requested moment. "At least 4 animals, each bigger than the last, then the whale" means 4 animals and THEN
  the whale (the whale is not one of the 4).
- Fit the youngest listener named in the context: short sentences, concrete images, gentle repetition, a little humor.
- Nothing frightening: no real danger, no villains, no injuries. Sadness or worry is fine when it is resolved.
  Grown-ups are with the children near deep water.
- The story must be complete: beginning, middle and end, ending with a happy resolution, hope, or a gentle teaching
  moment a child can feel (show it, then say it once in one simple sentence).
- Use the children's names and ages from the context. Parents can be "Mom" and "Dad" unless named.
- Length: 500 to 900 words, in short paragraphs.
- Write in {language}.

Answer with the title on the first line, an empty line, then the story. No markdown, no headings, nothing else."""

BEATS_SYSTEM = """You turn a parent's story idea into a numbered checklist of the moments the story must contain.

- One item per distinct event, in order. Keep the parent's details in the item (which animal, what is seen, how
  something happens, e.g. "the whale expels the rock through her blowhole and it shoots into the sky").
- Keep counts as their own item ("at least 4 different sea animals, each bigger than the last, before the whale").
  A character who comes after "the last one" (like the whale) is not part of that count.
- Include moments of feeling the parent asks for (the children are sad and think they were tricked).
- When the parent describes a picture ("we see an illustration of..."), make it an item too.
- Don't add anything the parent didn't ask for. Write in English."""

BEATS_SCHEMA = {
    "type": "object",
    "properties": {"beats": {"type": "array", "items": {"type": "string"}}},
    "required": ["beats"],
}

CRITIC_SYSTEM = """You are a careful children's-book editor. Check a draft story against the parent's checklist.

1. For EACH checklist item, in order: copy the sentence(s) of the draft that show it as "evidence" (exact quote, or ""
   if there is none), then say if it is fully present, with every detail and count of the item. Count animals or
   steps one by one in your note. Partial or merged counts as missing.
2. Then read the draft as a strict editor and list concrete problems:
   - every missing or incomplete checklist item;
   - time and place contradictions (seeing the night sky in the morning, being home and at the beach at once);
   - unsafe examples for a small child (young children alone near deep water); events without a reason; characters
     who appear or vanish; a rushed ending; a lesson repeated more than once.
   NEVER ask to remove, soften or replace something the checklist asks for, even if it seems odd: the parent wants it.
   Don't list matters of taste.
3. Then the checks. verdict is "pass" only if every item is present, every check is true and there is no problem."""

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "beat": {"type": "string"},
                    "evidence": {"type": "string"},
                    "note": {"type": "string"},
                    "present": {"type": "boolean"},
                },
                "required": ["beat", "evidence", "note", "present"],
            },
        },
        "problems": {"type": "array", "items": {"type": "string"}},
        "complete_arc": {"type": "boolean"},
        "happy_or_hopeful_ending": {"type": "boolean"},
        "teaching_moment": {"type": "string"},
        "age_appropriate": {"type": "boolean"},
        "consistent": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["pass", "revise"]},
    },
    "required": [
        "beats",
        "problems",
        "complete_arc",
        "happy_or_hopeful_ending",
        "teaching_moment",
        "age_appropriate",
        "consistent",
        "verdict",
    ],
}

REVISE_INSTRUCTION = """Revise the story to fix these problems. Keep everything that already works and every checklist
moment, keep the same language and length, and answer with the full revised story in the same format (title, empty
line, story).

Problems:
{problems}"""

BIBLE_SYSTEM = """You are the art director of a children's picture book. From the story and the family context, write
a visual bible so every illustration looks consistent. In English.

- style: one paragraph describing a single, soft, warm illustration style for young children (medium, palette,
  lighting, line quality). Same style for every page.
- characters: everyone who appears in a picture (children, parents, every animal the children meet). For each: a
  short id (lowercase, no spaces), name, role, apparent age, and "look": a precise, unchanging description (body size
  for the age, hair color and style, eye color, skin tone, ONE fixed outfit with colors, any accessory; for animals
  species, size, colors, markings). Toddlers must look like toddlers. Do not describe actions.
- props: important recurring objects (id, name, look), e.g. a magic object, a boat. The look starts with its real
  size compared to a child ("small enough to fit in a toddler's palm")."""

BIBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "style": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "age": {"type": "string"},
                    "look": {"type": "string"},
                },
                "required": ["id", "name", "role", "age", "look"],
            },
        },
        "props": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "name": {"type": "string"}, "look": {"type": "string"}},
                "required": ["id", "name", "look"],
            },
        },
    },
    "required": ["style", "characters", "props"],
}

PAGES_SYSTEM = """You turn a finished story into the pages of a picture book. Each page has two parts that work
together: the text read aloud, and one illustration.

- Use between 10 and 18 pages. Follow the story in order and cover all of it; the last page shows the ending.
- Every checklist moment gets at least one page. "moments" lists the checklist numbers a page shows.
- One moment per picture: never a split scene, montage or "then... and then". When the story repeats a pattern (asking
  one animal after another), give each step its own page so the child sees it grow.
- text: 1 to 3 short sentences taken or condensed from the story, in {language}. Reading every page's text in order
  must tell the complete story on its own.
- illustration: in English, ONE clear moment: who is there, what they do, their expressions, where, time of day and
  light. It shows what the text doesn't say (feelings, scenery, details) and never contradicts it (same time of day,
  same place). Name characters by name. No text, letters or speech bubbles in the image.
- characters: ids (from the bible) of the characters visible, the most important for this page first (the animal or
  character the page is about comes before the family).
- props: ids of the props visible.
- setting: short phrase (place + time of day)."""

PAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "moments": {"type": "array", "items": {"type": "integer"}},
        "text": {"type": "string"},
        "illustration": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "string"}},
        "props": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
    },
    "required": ["moments", "text", "illustration", "characters", "props", "setting"],
}
PAGES_SCHEMA = {
    "type": "object",
    "properties": {"pages": {"type": "array", "items": PAGE_SCHEMA}},
    "required": ["pages"],
}

PAGES_CHECK_SYSTEM = """You review picture-book pages against the full story and the parent's checklist. Fix any page
where: an event of the story is missing or out of order; text and illustration contradict each other (time of day,
place, who is there); the characters list misses someone visible or lists someone absent, or the main character of the
page isn't first; an illustration shows more than one moment (split scene, montage); the story doesn't progress
clearly from one page to the next; the last page isn't the ending. Every checklist moment must be on a page: add pages
where one is missing. Return the complete corrected list of pages (unchanged pages as they were)."""

PAGES_FIX = """These checklist moments are on no page yet:
{missing}

Return the complete list of pages again, with new pages added in the right places for each of them (and their
"moments" set)."""


def user_request(context: str, prompt: str) -> str:
    return f"Family context:\n{context.strip() or '(none)'}\n\nThe parent's story idea:\n{prompt.strip()}"


def checklist(beats: list[str]) -> str:
    return "\n".join(f"{i}. {b}" for i, b in enumerate(beats, 1))


def parse_story(text: str) -> tuple[str, str]:
    """Title on the first line, the story after. Tolerates "Title:", markdown and quotes."""
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return "", ""
    title = re.sub(r"^(#+\s*|\*+|title\s*:\s*)", "", lines[0].strip(), flags=re.I).strip("*\"' ")
    body = "\n".join(lines[1:]).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)
    return title, body


def looks_broken(story: str) -> str | None:
    """A draft that derailed (too short, or a run of numbers/repeats) is retried rather than reviewed."""
    words = story.split()
    if len(words) < 250:
        return f"too short ({len(words)} words)"
    if re.search(r"(\b\d+\.\s*){8,}", story):
        return "a run of numbers"
    if len(set(words[-80:])) < 15:
        return "repeating itself"
    return None


def image_prompt(style: str, illustration: str, setting: str, chars: list[dict], props: list[dict]) -> str:
    parts = [f"Children's picture book illustration. {style}", f"Scene: {illustration}", f"Setting: {setting}."]
    if chars:
        parts.append(
            "Characters (match the reference images): "
            + " ".join(f"{c['name']} ({c['age']}): {c['look']}." for c in chars)
        )
    if props:
        parts.append("Objects: " + " ".join(f"{p['name']}: {p['look']}." for p in props))
    parts.append(
        "Each character appears only once. One single scene, no panels. No text, no letters, no speech bubbles."
    )
    return "\n".join(parts)


def ref_prompt(style: str, char: dict) -> str:
    return (
        f"Children's picture book character reference. {style}\n"
        f"Full-body portrait of {char['name']}, {char['role']}, {char['age']}: {char['look']}\n"
        "Standing, facing the viewer, gentle smile, whole body visible, plain white background, only this one "
        "character. No text, no letters."
    )
