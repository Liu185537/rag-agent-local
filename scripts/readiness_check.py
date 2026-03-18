from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import requests


def check(cond: bool, name: str, detail: str) -> dict[str, Any]:
    """构造单项检查结果结构。"""
    return {"name": name, "ok": bool(cond), "detail": detail}


def main() -> None:
    """面试前就绪检查。

    覆盖两大类：
    1. 关键文件是否存在；
    2. 核心接口是否可访问。
    """
    parser = argparse.ArgumentParser(description="RAG-Agent 面试就绪检查脚本。")
    parser.add_argument("--base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--namespace", default="demo")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    checks: list[dict[str, Any]] = []

    # 先检查仓库关键文件。
    required_files = [
        "app/main.py",
        "app/templates/playground.html",
        "scripts/interview_demo.py",
        "scripts/run_eval.py",
        ".github/workflows/ci.yml",
        "README.md",
    ]
    for rel in required_files:
        p = project_root / rel
        checks.append(check(p.exists(), f"file:{rel}", str(p)))

    # 再检查服务接口可用性。
    try:
        health = requests.get(f"{args.base_url.rstrip('/')}/api/v1/health", timeout=10)
        checks.append(check(health.status_code == 200, "api:health", f"status={health.status_code}"))
    except Exception as exc:
        checks.append(check(False, "api:health", str(exc)))

    try:
        pg = requests.get(f"{args.base_url.rstrip('/')}/playground", timeout=10)
        checks.append(check(pg.status_code == 200, "api:playground", f"status={pg.status_code}"))
    except Exception as exc:
        checks.append(check(False, "api:playground", str(exc)))

    try:
        demo_summary = requests.get(
            f"{args.base_url.rstrip('/')}/api/v1/demo/summary",
            params={"namespace": args.namespace},
            timeout=10,
        )
        ok = demo_summary.status_code == 200 and "stats" in demo_summary.json()
        checks.append(check(ok, "api:demo_summary", f"status={demo_summary.status_code}"))
    except Exception as exc:
        checks.append(check(False, "api:demo_summary", str(exc)))

    passed = sum(1 for item in checks if item["ok"])
    total = len(checks)
    print("=== 面试就绪检查 ===")
    for item in checks:
        mark = "通过" if item["ok"] else "失败"
        print(f"[{mark}] {item['name']} - {item['detail']}")
    print(f"\n结果：{passed}/{total} 项通过")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
