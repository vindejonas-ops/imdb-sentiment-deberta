from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "imdb_top_500.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

SOURCE_MODEL_ID = "textattack/roberta-base-imdb"
TARGET_REPO_ID = "Ch1kage/Sentiment_analyse-HuggingFace"
MIN_ACCURACY = 0.92
BATCH_SIZE = 8
MAX_LENGTH = 512


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 加载数据
    df = pd.read_csv(DATA_PATH)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    print(f"Loading model: {SOURCE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(SOURCE_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(SOURCE_MODEL_ID)
    model.eval()

    # 批量推理
    predictions = []
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            batch_preds = logits.argmax(dim=-1).tolist()
            predictions.extend(batch_preds)
            print(f"Evaluated {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")

    # 计算指标
    accuracy = accuracy_score(labels, predictions)
    correct = sum(1 for l, p in zip(labels, predictions) if l == p)
    report = classification_report(
        labels, predictions, target_names=["negative", "positive"], output_dict=True
    )

    metrics = {
        "source_model": SOURCE_MODEL_ID,
        "target_repo": TARGET_REPO_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "total": len(labels),
        "correct": correct,
        "accuracy": float(accuracy),
        "min_accuracy": MIN_ACCURACY,
        "classification_report": report,
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    # 保存模型和tokenizer用于上传
    tokenizer.save_pretrained(ARTIFACTS_DIR)
    model.save_pretrained(ARTIFACTS_DIR, safe_serialization=True)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{len(labels)}")

    if accuracy < MIN_ACCURACY:
        raise SystemExit(f"Accuracy {accuracy:.4f} below minimum {MIN_ACCURACY}")


if __name__ == "__main__":
    main()