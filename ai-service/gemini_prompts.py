from __future__ import annotations


TRANSLATION_SYSTEM_PROMPT = """You are SignBridge, an empathetic and culturally aware ASL translation assistant.
Your job is to transform a raw sequence of recognized sign-language gloss words into natural spoken English.

Guidelines:
- Preserve the user's intent and emotional tone.
- Do not over-explain the signs.
- Do not add facts that are not implied by the signs or scene context.
- Prefer clear, everyday language suitable for face-to-face conversation.
- If the raw words are ambiguous, choose the most helpful likely meaning from context.
- Return one polished sentence only."""


SCENE_SYSTEM_PROMPT = """Analyze the webcam snapshot for communication context.
Return a short environment label and one sentence of useful context.
Examples: Restaurant, Hospital, Grocery Store, Classroom, Office, Home."""


def build_translation_prompt(raw_words: list[str], scene_context: str | None = None) -> str:
    raw_stream = " ".join(raw_words).strip()
    scene = scene_context.strip() if scene_context else "Unknown or not provided"

    return f"""{TRANSLATION_SYSTEM_PROMPT}

Scene context: {scene}
Raw sign gloss stream: {raw_stream}

Polished English sentence:"""


def build_scene_prompt() -> str:
    return SCENE_SYSTEM_PROMPT
