from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


def request_json(method: str, url: str, **kwargs) -> tuple[int, dict[str, Any]]:
    """统一 HTTP 请求入口，返回 (状态码, JSON/原始文本包装)。"""
    resp = requests.request(method, url, timeout=120, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}
    return resp.status_code, data


def main() -> None:
    """执行端到端面试演示并产出报告。

    流程概览：
    1. 健康检查；
    2. 写入 Agent 配置；
    3. （可选）灌入样例知识；
    4. 依次发起多轮问答与检索；
    5. 拉取汇总信息并生成 JSON/Markdown 报告。
    """
    parser = argparse.ArgumentParser(description="运行端到端面试演示，并自动生成报告。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--namespace", default="interview_demo")
    parser.add_argument("--report-dir", default="eval/reports")
    parser.add_argument("--skip-ingest", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    namespace = args.namespace

    project_root = Path(__file__).resolve().parents[1]
    report_dir = project_root / args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    # timeline 记录每个步骤的状态，便于回放演示过程。
    timeline: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    code, health = request_json("GET", f"{base_url}/api/v1/health")
    timeline.append({"step": "health", "status_code": code, "ok": code == 200, "data": health})
    if code != 200:
        raise SystemExit(f"健康检查失败：{health}")

    # 设置本次演示使用的 Agent 人设与回答约束。
    cfg_payload = {
        "namespace": namespace,
        "name": "面试销售教练",
        "description": "用于演示 RAG 工作流的面试 Agent",
        "instructions": (
            "请用中文作答，结构简洁。"
            "引用检索到的上下文时，请标注如 [1] 的引用标记。"
        ),
    }
    code, cfg = request_json("PUT", f"{base_url}/api/v1/agent-config", json=cfg_payload)
    timeline.append({"step": "save_agent_config", "status_code": code, "ok": code == 200, "data": cfg})
    if code != 200:
        raise SystemExit(f"保存 Agent 配置失败：{cfg}")

    if not args.skip_ingest:
        # 按需入库样例知识，确保演示环境可重复。
        sample_file = project_root / "data" / "sample_knowledge.md"
        text = sample_file.read_text(encoding="utf-8")
        ingest_payload = {
            "namespace": namespace,
            "docs": [
                {
                    "doc_id": "sample_marketing_playbook",
                    "source": "sample_knowledge.md",
                    "text": text,
                    "metadata": {"topic": "education_marketing", "scenario": "interview_demo"},
                }
            ],
        }
        code, ingest = request_json("POST", f"{base_url}/api/v1/knowledge/ingest", json=ingest_payload)
        timeline.append({"step": "ingest_sample_doc", "status_code": code, "ok": code == 200, "data": ingest})
        if code != 200:
            raise SystemExit(f"知识摄取失败：{ingest}")

    questions = [
        "家长说价格太高，我第一句怎么回复更稳妥？",
        "对方说最近忙，怎么做三步跟进最自然？",
        "如何做不冒犯的需求资格判断？",
    ]
    qa_results: list[dict[str, Any]] = []
    session_id: str | None = None
    for q in questions:
        # 复用 session_id，模拟同一用户连续对话。
        payload = {"message": q, "namespace": namespace}
        if session_id:
            payload["session_id"] = session_id
        code, chat_data = request_json("POST", f"{base_url}/api/v1/chat", json=payload)
        ok = code == 200
        timeline.append({"step": "chat", "question": q, "status_code": code, "ok": ok})
        if not ok:
            raise SystemExit(f"问题 `{q}` 的对话请求失败：{chat_data}")

        session_id = chat_data.get("session_id", session_id)
        citations = chat_data.get("citations", [])
        qa_results.append(
            {
                "question": q,
                "answer_preview": (chat_data.get("answer") or "")[:260],
                "citation_count": len(citations),
                "session_id": session_id,
            }
        )

        # 同步记录纯检索结果，验证召回文档是否合理。
        code_r, retrieve_data = request_json(
            "POST",
            f"{base_url}/api/v1/retrieve",
            json={"query": q, "namespace": namespace, "top_k": 3},
        )
        timeline.append(
            {
                "step": "retrieve",
                "question": q,
                "status_code": code_r,
                "ok": code_r == 200,
                "top_result_doc_id": (
                    retrieve_data.get("chunks", [{}])[0].get("doc_id")
                    if retrieve_data.get("chunks")
                    else None
                ),
            }
        )

    code, summary = request_json("GET", f"{base_url}/api/v1/demo/summary", params={"namespace": namespace})
    timeline.append({"step": "demo_summary", "status_code": code, "ok": code == 200})
    if code != 200:
        raise SystemExit(f"演示摘要获取失败：{summary}")

    # 汇总执行表现。
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pass_rate = sum(1 for x in timeline if x.get("ok")) / len(timeline) if timeline else 0.0

    output = {
        "base_url": base_url,
        "namespace": namespace,
        "elapsed_ms": round(elapsed_ms, 2),
        "timeline": timeline,
        "qa_results": qa_results,
        "summary": summary,
        "pass_rate": round(pass_rate, 4),
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"interview_demo_{ts}.json"
    md_path = report_dir / f"interview_demo_{ts}.md"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成便于面试讲解的 Markdown 摘要。
    lines = [
        "# 面试演示报告",
        "",
        f"- 时间戳：{ts}",
        f"- 服务地址：`{base_url}`",
        f"- 命名空间：`{namespace}`",
        f"- 总耗时：`{round(elapsed_ms, 2)} ms`",
        f"- 通过率：`{round(pass_rate * 100, 2)}%`",
        "",
        "## 问答结果",
        "",
    ]
    for i, item in enumerate(qa_results, start=1):
        lines.extend(
            [
                f"### Q{i}",
                f"- 问题：{item['question']}",
                f"- 引用数量：{item['citation_count']}",
                f"- 回答预览：{item['answer_preview']}",
                "",
            ]
        )

    stats = summary.get("stats", {})
    lines.extend(
        [
            "## 命名空间统计",
            "",
            f"- 文档数：{stats.get('doc_count', 0)}",
            f"- 分块数：{stats.get('chunk_count', 0)}",
            f"- 会话数：{stats.get('session_count', 0)}",
            "",
            "## 产出文件",
            "",
            f"- JSON 报告：`{json_path}`",
            f"- Markdown 报告：`{md_path}`",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("面试演示执行完成。")
    print(f"通过率：{round(pass_rate * 100, 2)}%")
    print(f"JSON 报告：{json_path}")
    print(f"Markdown 报告：{md_path}")


if __name__ == "__main__":
    main()
