import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def load_id2label(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def predict(text: str, model_dir: str = "outputs/android", max_length: int = 64):
    model_dir = Path(model_dir)

    onnx_path = model_dir / "camera_bert.onnx"
    id2label_path = model_dir / "id2label.json"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if not id2label_path.exists():
        raise FileNotFoundError(f"id2label.json not found: {id2label_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    id2label = load_id2label(id2label_path)

    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    if "token_type_ids" in encoded:
        token_type_ids = encoded["token_type_ids"].astype(np.int64)
    else:
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    outputs = session.run(
        ["logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    logits = outputs[0]
    probs = softmax(logits)

    pred_id = int(np.argmax(probs, axis=-1)[0])
    confidence = float(probs[0][pred_id])

    return {
        "text": text,
        "label": id2label[pred_id],
        "confidence": round(confidence, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="outputs/android")
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
