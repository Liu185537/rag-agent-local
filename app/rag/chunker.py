from __future__ import annotations


def chunk_text(text: str, max_chars: int = 600, overlap_chars: int = 80) -> list[str]:
    """将长文本切分为可检索的分块。

    策略说明：
    1. 先按段落（空行）切分，尽量保持语义完整；
    2. 超长段落使用滑动窗口拆分；
    3. 窗口之间保留 overlap，减少信息截断。
    """
    normalized = "\n".join(line.strip() for line in text.splitlines()).strip()
    if not normalized:
        return []

    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # 单段超过上限时，直接走窗口切分。
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

        # 未超长则尽量与当前块拼接，减少碎片。
        projected = len(current) + len(paragraph) + (2 if current else 0)
        if projected <= max_chars:
            current = f"{current}\n\n{paragraph}" if current else paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current:
        chunks.append(current.strip())
    return chunks

