"""Train and evaluate a SciBERT citation-intent student classifier.

The input label file can be:
- a Parquet file with `label` plus citing/cited title and abstract columns, or
- the legacy JSON checkpoint from `deliverable_2.ipynb` with `text1`, `text2`,
  and numeric labels.

Usage:
    PYTHONPATH=src python -m citation_cartels.annotation.train_student \
        --config configs/cikm_middle.yaml \
        --labels data/labels/teacher_labels_final.parquet
"""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_labels(path: Path, labels: list[str]) -> pd.DataFrame:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(data)
    else:
        frame = pd.read_parquet(path)

    if "label" not in frame.columns:
        raise ValueError("Label file must contain a `label` column.")

    if pd.api.types.is_numeric_dtype(frame["label"]):
        frame["label_name"] = frame["label"].map({idx: label for idx, label in enumerate(labels)})
    else:
        frame["label_name"] = frame["label"].astype(str)

    if {"text1", "text2"}.issubset(frame.columns):
        frame["citing_text"] = frame["text1"].fillna("")
        frame["cited_text"] = frame["text2"].fillna("")
    else:
        required = ["citing_title", "citing_abstract", "cited_title", "cited_abstract"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Missing text columns for student training: {missing}")
        frame["citing_text"] = (frame["citing_title"].fillna("") + ". " + frame["citing_abstract"].fillna("")).str.strip()
        frame["cited_text"] = (frame["cited_title"].fillna("") + ". " + frame["cited_abstract"].fillna("")).str.strip()

    frame = frame[frame["label_name"].isin(labels)].copy()
    if frame.empty:
        raise ValueError("No valid labels remain after filtering to configured label set.")
    return frame[["citing_text", "cited_text", "label_name"]]


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["citing_text"],
            batch["cited_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(tokenize, batched=True)


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **_: Any,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class ProgressCallback(TrainerCallback):
    def __init__(self, report_dir: Path, total_train_examples: int, effective_batch_size: int) -> None:
        self.progress_path = report_dir / "scibert_training_progress.jsonl"
        self.total_train_examples = total_train_examples
        self.effective_batch_size = effective_batch_size
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args: TrainingArguments, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any):
        logs = logs or {}
        max_steps = state.max_steps or 0
        examples_seen = min(int(state.global_step * self.effective_batch_size), self.total_train_examples * int(args.num_train_epochs))
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_step": int(state.global_step),
            "max_steps": int(max_steps),
            "progress_pct": round((state.global_step / max_steps * 100), 2) if max_steps else None,
            "epoch": round(float(state.epoch), 4) if state.epoch is not None else None,
            "estimated_examples_seen": examples_seen,
            "effective_batch_size": self.effective_batch_size,
            "logs": {key: float(value) if isinstance(value, (int, float, np.floating)) else value for key, value in logs.items()},
        }
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")


def validate_label_counts(frame: pd.DataFrame, labels: list[str], min_count: int) -> None:
    counts = frame["label_name"].value_counts().reindex(labels, fill_value=0)
    weak = counts[counts < min_count]
    if not weak.empty:
        details = ", ".join(f"{label}={int(count)}" for label, count in weak.items())
        raise ValueError(
            "Not enough examples for reliable stratified SciBERT training. "
            f"Minimum required per label is {min_count}; weak labels: {details}"
        )


def limit_examples(frame: pd.DataFrame, max_examples: Optional[int], seed: int) -> pd.DataFrame:
    if max_examples is None or len(frame) <= max_examples:
        return frame
    label_counts = frame["label_name"].value_counts()
    fractions = label_counts / len(frame)
    per_label = (fractions * max_examples).round().astype(int).clip(lower=1)
    sampled = []
    for label, count in per_label.items():
        label_frame = frame[frame["label_name"] == label]
        sampled.append(label_frame.sample(min(int(count), len(label_frame)), random_state=seed))
    return pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed)


def build_training_args(
    output_dir: Path,
    config: dict[str, Any],
    seed: int,
    epochs: Optional[int],
    learning_rate: float,
    train_batch_size: int,
    eval_batch_size: int,
    gradient_accumulation_steps: int,
    save_steps: Optional[int],
    eval_strategy: str,
    logging_steps: int,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": epochs or int(config["student_model"]["epochs"]),
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_strategy": "steps",
        "logging_steps": logging_steps,
        "fp16": torch.cuda.is_available(),
        "report_to": "none",
        "seed": seed,
    }
    strategy_arg = "evaluation_strategy"
    if "evaluation_strategy" not in inspect.signature(TrainingArguments).parameters:
        strategy_arg = "eval_strategy"
    kwargs[strategy_arg] = eval_strategy
    if save_steps is not None:
        kwargs["save_strategy"] = "steps"
        kwargs["save_steps"] = save_steps
        if eval_strategy == "steps":
            kwargs["eval_steps"] = save_steps
    return TrainingArguments(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cikm_middle.yaml", type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output-dir", default="models/scibert_citation_intent", type=Path)
    parser.add_argument("--report-dir", default="reports", type=Path)
    parser.add_argument("--min-count-per-label", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--eval-strategy", choices=["epoch", "steps", "no"], default="epoch")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    labels = list(config["annotation"]["labels"])
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    seed = int(config["project"]["seed"])

    frame = load_labels(args.labels, labels)
    frame = limit_examples(frame, args.max_examples, seed)
    validate_label_counts(frame, labels, args.min_count_per_label)
    frame["labels"] = frame["label_name"].map(label_to_id)

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.2,
        random_state=seed,
        stratify=frame["labels"],
    )
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.5,
        random_state=seed,
        stratify=temp_frame["labels"],
    )

    model_name = config["student_model"]["base_model"]
    max_length = int(config["student_model"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = tokenize_dataset(Dataset.from_pandas(train_frame, preserve_index=False), tokenizer, max_length)
    validation_dataset = tokenize_dataset(Dataset.from_pandas(validation_frame, preserve_index=False), tokenizer, max_length)
    test_dataset = tokenize_dataset(Dataset.from_pandas(test_frame, preserve_index=False), tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    label_counts = np.bincount(train_frame["labels"].to_numpy(), minlength=len(labels))
    class_weights = len(train_frame) / (len(labels) * np.maximum(label_counts, 1))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, gold = eval_pred
        pred = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(gold, pred),
            "macro_f1": f1_score(gold, pred, average="macro"),
            "weighted_f1": f1_score(gold, pred, average="weighted"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_args(
        args.output_dir,
        config,
        seed,
        args.epochs,
        args.learning_rate,
        args.train_batch_size,
        args.eval_batch_size,
        args.gradient_accumulation_steps,
        args.save_steps,
        args.eval_strategy,
        args.logging_steps,
    )
    effective_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback(report_dir, len(train_frame), effective_batch_size)],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    test_metrics = trainer.evaluate(test_dataset)
    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    gold_ids = predictions.label_ids

    report = classification_report(
        gold_ids,
        pred_ids,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        gold_ids,
        pred_ids,
        target_names=labels,
        zero_division=0,
    )
    matrix = confusion_matrix(gold_ids, pred_ids, labels=list(range(len(labels))))

    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    eval_payload = {
        "labels_path": str(args.labels),
        "model_name": model_name,
        "num_train": int(len(train_frame)),
        "num_validation": int(len(validation_frame)),
        "num_test": int(len(test_frame)),
        "test_metrics": test_metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    (report_dir / "scibert_eval.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
    (report_dir / "scibert_classification_report.md").write_text(
        "```text\n" + report_text + "\n```\n",
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
    fig.tight_layout()
    fig.savefig(report_dir / "scibert_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(json.dumps(test_metrics, indent=2))
    print(f"Saved model to {args.output_dir / 'best_model'}")
    print(f"Saved reports to {report_dir}")


if __name__ == "__main__":
    main()
