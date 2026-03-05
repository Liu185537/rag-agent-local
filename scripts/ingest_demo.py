from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest sample knowledge into local RAG-Agent.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--namespace", default="demo")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    sample_file = project_root / "data" / "sample_knowledge.md"
    text = sample_file.read_text(encoding="utf-8")

    payload = {
        "namespace": args.namespace,
        "docs": [
            {
                "doc_id": "sample_marketing_playbook",
                "source": "sample_knowledge.md",
                "text": text,
                "metadata": {"topic": "education_marketing"},
            }
        ],
    }

    resp = requests.post(f"{args.base_url}/api/v1/knowledge/ingest", json=payload, timeout=120)
    print(f"status={resp.status_code}")
    print(resp.text)


if __name__ == "__main__":
    main()

