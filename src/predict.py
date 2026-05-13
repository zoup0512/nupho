import argparse
import json
from pathlib import Path
import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_id2label(model_dir: Path, model):
    id2label_path = model_dir / "id2label.json"

    if id2label_path.exists():
        with open(id2label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}

    return {int(k): v for k, v in model.config.id2label.items()}


def predict(text: str, model_dir: str = "outputs/bert", max_length: int = 64):
    model_path = Path(model_dir)

    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    id2label = load_id2label(model_path, model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - load_start

    infer_start = time.perf_counter()
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    confidence, pred_id = torch.max(probs, dim=-1)

    pred_id = pred_id.item()
    confidence = confidence.item()
    inference_time = time.perf_counter() - infer_start

    return {
        "text": text,
        "label": id2label[pred_id],
        "confidence": round(confidence, 4),
        "load_time_sec": round(load_time, 4),
        "inference_time_sec": round(inference_time, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="outputs/bert")
    parser.add_argument("--max_length", type=int, default=64)
    args = parser.parse_args()

    result = predict(
        text=args.text,
        model_dir=args.model_dir,
        max_length=args.max_length,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
