from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import requests


def load_dataset(dataset_path: Path) -> list[dict]:
    rows = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation for local RAG-Agent with report output.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--namespace", default="demo")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--report-dir", default="eval/reports")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = project_root / args.dataset
    report_dir = project_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)

    total = len(data)
    if total == 0:
        raise SystemExit("Dataset is empty.")

    keyword_hits = 0
    citation_hits = 0
    retrieval_hits = 0
    citation_doc_hits = 0
    latencies_ms: list[float] = []
    details: list[dict] = []

    for idx, row in enumerate(data, start=1):
        question = row["question"]
        expected_keywords = [kw.lower() for kw in row.get("expected_keywords", [])]
        expected_doc_id = row.get("expected_doc_id")

        chat_payload = {
            "message": question,
            "namespace": args.namespace,
        }
        start_chat = time.perf_counter()
        chat_resp = requests.post(f"{args.base_url}/api/v1/chat", json=chat_payload, timeout=120)
        latency_ms = (time.perf_counter() - start_chat) * 1000.0
        latencies_ms.append(latency_ms)
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()
        answer = chat_data["answer"].lower()
        citations = chat_data.get("citations", [])
        citation_doc_ids = {item.get("doc_id") for item in citations}

        keyword_hit = bool(expected_keywords and any(kw in answer for kw in expected_keywords))
        citation_hit = bool(citations)
        citation_doc_hit = bool(expected_doc_id and expected_doc_id in citation_doc_ids)

        if keyword_hit:
            keyword_hits += 1
        if citation_hit:
            citation_hits += 1
        if citation_doc_hit:
            citation_doc_hits += 1

        retrieve_payload = {"query": question, "namespace": args.namespace, "top_k": args.top_k}
        retrieve_resp = requests.post(
            f"{args.base_url}/api/v1/retrieve", json=retrieve_payload, timeout=60
        )
        retrieve_resp.raise_for_status()
        retrieve_data = retrieve_resp.json()
        retrieved_doc_ids = {item["doc_id"] for item in retrieve_data.get("chunks", [])}
        retrieval_hit = bool(expected_doc_id and expected_doc_id in retrieved_doc_ids)
        if retrieval_hit:
            retrieval_hits += 1

        print(
            f"[{idx}/{total}] "
            f"keyword_hit={keyword_hit} "
            f"citation_hit={citation_hit} citation_doc_hit={citation_doc_hit} "
            f"retrieval_hit={retrieval_hit} latency_ms={latency_ms:.1f}"
        )
        details.append(
            {
                "idx": idx,
                "question": question,
                "expected_doc_id": expected_doc_id,
                "keyword_hit": keyword_hit,
                "citation_hit": citation_hit,
                "citation_doc_hit": citation_doc_hit,
                "retrieval_hit": retrieval_hit,
                "latency_ms": round(latency_ms, 3),
                "citation_count": len(citations),
                "answer_preview": answer[:200],
            }
        )

    def pct(x: int) -> float:
        return (x / total * 100.0) if total else 0.0

    latencies_sorted = sorted(latencies_ms)
    p50 = latencies_sorted[int(0.5 * (len(latencies_sorted) - 1))]
    p90 = latencies_sorted[int(0.9 * (len(latencies_sorted) - 1))]
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]
    avg_latency = sum(latencies_ms) / len(latencies_ms)

    print("\n=== Evaluation Summary ===")
    print(f"Samples: {total}")
    print(f"Keyword hit rate: {pct(keyword_hits):.2f}% ({keyword_hits}/{total})")
    print(f"Citation coverage: {pct(citation_hits):.2f}% ({citation_hits}/{total})")
    print(f"Citation doc hit rate: {pct(citation_doc_hits):.2f}% ({citation_doc_hits}/{total})")
    print(f"Retrieval hit@{args.top_k}: {pct(retrieval_hits):.2f}% ({retrieval_hits}/{total})")
    print(f"Latency avg/p50/p90/p95 (ms): {avg_latency:.2f}/{p50:.2f}/{p90:.2f}/{p95:.2f}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = report_dir / f"eval_report_{ts}.json"
    report_csv = report_dir / f"eval_report_{ts}.csv"
    summary = {
        "dataset": str(dataset_path),
        "base_url": args.base_url,
        "namespace": args.namespace,
        "samples": total,
        "keyword_hit_rate": round(pct(keyword_hits), 4),
        "citation_coverage": round(pct(citation_hits), 4),
        "citation_doc_hit_rate": round(pct(citation_doc_hits), 4),
        "retrieval_hit_rate": round(pct(retrieval_hits), 4),
        "retrieval_top_k": args.top_k,
        "latency_ms": {
            "avg": round(avg_latency, 3),
            "p50": round(p50, 3),
            "p90": round(p90, 3),
            "p95": round(p95, 3),
        },
    }
    report_payload = {"summary": summary, "details": details}
    report_json.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    print(f"Report JSON: {report_json}")
    print(f"Report CSV : {report_csv}")


if __name__ == "__main__":
    main()
