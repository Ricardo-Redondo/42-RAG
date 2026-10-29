import re

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
WORD_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def expand_identifiers(text: str) -> str:
    """
    Add the parts of snake_case / CamelCase names to the text.

    BM25 only matches exact words: a question saying "max model length"
    never matches `max_model_len`. The pieces are appended, the original
    text is kept, so exact names still match too:

        "uses max_model_len" -> "uses max_model_len max model len"

    Used on BOTH sides (chunks when indexing, questions when searching),
    so the two always go through the same processing.
    """
    parts: list[str] = []
    for name in IDENTIFIER.findall(text):
        if "_" in name or any(c.isupper() for c in name[1:]):
            for piece in name.split("_"):
                parts += WORD_PART.findall(piece)
    return text + " " + " ".join(parts)
