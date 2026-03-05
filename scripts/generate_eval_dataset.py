from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_rows(size: int) -> list[dict]:
    slots = [
        {
            "theme": "price objection",
            "patterns": [
                "How do I handle {theme} in first conversation?",
                "Parent says tuition is high, what should I reply?",
                "What is a short script for {theme} with empathy?",
                "How to convert a lead with {theme} into trial class booking?",
            ],
            "keywords": ["value", "trial", "budget"],
        },
        {
            "theme": "follow-up cadence",
            "patterns": [
                "What follow-up cadence should I use after busy reply?",
                "Can you give a 3-step {theme} plan?",
                "How often should I message a lead who said maybe later?",
                "Best reminder timing for enrollment decision?",
            ],
            "keywords": ["follow-up", "48 hours", "next step"],
        },
        {
            "theme": "qualification question",
            "patterns": [
                "How to ask {theme} without sounding pushy?",
                "What questions should I ask before giving recommendation?",
                "How to discover parent goals in first chat?",
                "What is a friendly qualification script?",
            ],
            "keywords": ["open question", "goal", "empathy"],
        },
    ]

    rows: list[dict] = []
    i = 0
    while len(rows) < size:
        bucket = slots[i % len(slots)]
        pattern = bucket["patterns"][(i // len(slots)) % len(bucket["patterns"])]
        question = pattern.format(theme=bucket["theme"])
        rows.append(
            {
                "question": question,
                "expected_keywords": bucket["keywords"],
                "expected_doc_id": "sample_marketing_playbook",
            }
        )
        i += 1
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eval dataset for RAG-Agent.")
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--output", default="eval/dataset_v2.jsonl")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args.size)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generated {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

