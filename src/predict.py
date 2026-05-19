from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_MODEL = "textattack/roberta-base-imdb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict IMDB review sentiment.")
    parser.add_argument("text", help="Review text to classify.")
    parser.add_argument(
        "--model",
        default=str(ARTIFACTS_DIR) if ARTIFACTS_DIR.exists() else DEFAULT_MODEL,
        help="Local artifact directory or Hugging Face model ID.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    encoded = tokenizer(
        [args.text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)[0]
    label = int(probabilities.argmax().item())
    sentiment = "positive" if label == 1 else "negative"

    print(f"sentiment={sentiment}")
    print(f"negative_probability={float(probabilities[0]):.4f}")
    print(f"positive_probability={float(probabilities[1]):.4f}")


if __name__ == "__main__":
    main()