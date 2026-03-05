from __future__ import annotations


def chunk_text(text: str, max_chars: int = 600, overlap_chars: int = 80) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                window = paragraph[start:end].strip()
                if window:
                    chunks.append(window)
                if end == len(paragraph):
                    break
                start = end - overlap_chars
            continue

        projected = len(current) + len(paragraph) + (2 if current else 0)
        if projected <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())
    return chunks

