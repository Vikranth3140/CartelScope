"""Check or download an Azure OpenAI Batch API result.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.retrieve_azure_batch \
        --batch-id batch_...

When the batch is complete, add `--download-output` to write the output JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def client_from_env() -> Any:
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=require_env("AZURE_OPENAI_ENDPOINT"),
        api_key=require_env("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
    )


def batch_to_dict(batch: Any) -> dict[str, Any]:
    return {
        "id": batch.id,
        "status": getattr(batch, "status", None),
        "created_at": getattr(batch, "created_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expires_at": getattr(batch, "expires_at", None),
        "request_counts": getattr(batch, "request_counts", None),
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }


def download_file(client: Any, file_id: str, output_path: Path) -> None:
    content = client.files.content(file_id)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(content.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--download-output", action="store_true")
    parser.add_argument("--output", default=None, type=Path)
    parser.add_argument("--download-errors", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    labels_dir = Path(config["paths"]["labels_dir"])
    manifest_dir = Path(config["paths"]["manifest_dir"])
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    client = client_from_env()
    batch = client.batches.retrieve(args.batch_id)
    payload = batch_to_dict(batch)
    payload["checked_at"] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(payload, indent=2, default=str))

    manifest_path = manifest_dir / f"azure_teacher_batch_status_{args.batch_id}.json"
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    output_file_id: Optional[str] = payload.get("output_file_id")
    if args.download_output:
        if not output_file_id:
            raise RuntimeError(f"Batch {args.batch_id} has no output_file_id yet. Status: {payload['status']}")
        output_path = args.output or labels_dir / f"{args.batch_id}_output.jsonl"
        download_file(client, output_file_id, output_path)
        print(f"Downloaded output to {output_path}")

    error_file_id: Optional[str] = payload.get("error_file_id")
    if args.download_errors and error_file_id:
        error_path = labels_dir / f"{args.batch_id}_errors.jsonl"
        download_file(client, error_file_id, error_path)
        print(f"Downloaded errors to {error_path}")


if __name__ == "__main__":
    main()
