from __future__ import annotations

import argparse
from pathlib import Path

import requests


def main() -> None:
    """将示例知识文档写入指定 namespace。

    用法场景：本地启动服务后，快速准备可检索的演示数据。
    """
    parser = argparse.ArgumentParser(description="Ingest sample knowledge into local RAG-Agent.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--namespace", default="demo")
    args = parser.parse_args()

    # 读取项目内置示例知识。
    project_root = Path(__file__).resolve().parents[1]
    sample_file = project_root / "data" / "sample_knowledge.md"
    text = sample_file.read_text(encoding="utf-8")

    # 组装与后端 /knowledge/ingest 对应的请求体。
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

    # 调用 API 执行入库并打印结果。
    resp = requests.post(f"{args.base_url}/api/v1/knowledge/ingest", json=payload, timeout=120)
    print(f"status={resp.status_code}")
    print(resp.text)


if __name__ == "__main__":
    main()

