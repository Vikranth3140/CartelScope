"""Submit an Azure OpenAI teacher-labeling batch, dry-run by default.

This helper intentionally refuses to create paid usage unless `--execute` is
provided. It estimates cost from config and records a manifest for each run.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.submit_azure_batch \
        --config configs/cikm_middle.yaml \
        --batch-jsonl data/labels/teacher_batch_2000.jsonl

Paid execution, only after approval:
    PYTHONPATH=src python -m citation_cartels.annotation.submit_azure_batch \
        --config configs/cikm_middle.yaml \
        --batch-jsonl data/labels/teacher_batch_2000.jsonl \
        --execute
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_requests(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def estimate_cost(config: dict[str, Any], num_requests: int) -> dict[str, float]:
    estimate = config["azure"]["teacher_batch_estimate"]
    input_tokens = num_requests * float(estimate["input_tokens_per_edge"])
    output_tokens = num_requests * float(estimate["output_tokens_per_edge"])
    input_cost = input_tokens / 1_000_000 * float(estimate["batch_input_usd_per_1m_tokens"])
    output_cost = output_tokens / 1_000_000 * float(estimate["batch_output_usd_per_1m_tokens"])
    return {
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": output_tokens,
        "estimated_input_cost_usd": input_cost,
        "estimated_output_cost_usd": output_cost,
        "estimated_total_cost_usd": input_cost + output_cost,
    }


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def submit_batch(batch_jsonl: Path, endpoint: str, api_key: str) -> dict[str, Any]:
    # Import inside paid execution path so dry-run works without the package.
    from openai import AzureOpenAI

    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    with batch_jsonl.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/chat/completions",
        completion_window="24h",
    )

    return {
        "file_id": uploaded.id,
        "batch_id": batch.id,
        "status": getattr(batch, "status", None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--batch-jsonl", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_dir = Path(config["paths"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)

    if not args.batch_jsonl.exists():
        raise FileNotFoundError(args.batch_jsonl)

    num_requests = count_requests(args.batch_jsonl)
    cost = estimate_cost(config, num_requests)
    hard_cap = float(config["azure"]["hard_cap_usd"])

    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_jsonl": str(args.batch_jsonl),
        "num_requests": num_requests,
        "cost_estimate": cost,
        "hard_cap_usd": hard_cap,
        "executed": bool(args.execute),
    }

    print(json.dumps(manifest, indent=2))
    if cost["estimated_total_cost_usd"] > hard_cap:
        raise RuntimeError("Estimated batch cost exceeds configured hard cap.")

    if not args.execute:
        manifest_path = manifest_dir / "azure_teacher_batch_dry_run.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Dry run only. Wrote {manifest_path}. Add --execute only after approval.")
        return

    endpoint = require_env("AZURE_OPENAI_ENDPOINT")
    api_key = require_env("AZURE_OPENAI_API_KEY")
    result = submit_batch(args.batch_jsonl, endpoint=endpoint, api_key=api_key)
    manifest["azure_result"] = result

    manifest_path = manifest_dir / f"azure_teacher_batch_{result['batch_id']}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote execution manifest to {manifest_path}")


if __name__ == "__main__":
    main()
