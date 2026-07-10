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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict(text: str, model_dir: str = "outputs/android", max_length: int = 64):
    model_dir = Path(model_dir)

    onnx_path = model_dir / "bert.onnx"
    id2label_path = model_dir / "id2label.json"
    id2domain_path = model_dir / "id2domain.json"
    closure_id2label_path = model_dir / "closure_id2label.json"

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    id2label = {int(k): v for k, v in load_json(id2label_path).items()}
    id2domain = {int(k): v for k, v in load_json(id2domain_path).items()}
    closure_id2label = {
        int(k): v for k, v in load_json(closure_id2label_path).items()
    }

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
        ["domain_logits", "camera_logits", "closure_logits"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    domain_logits = outputs[0]
    camera_logits = outputs[1]
    closure_logits = outputs[2]

    domain_probs = softmax(domain_logits)
    domain_id = int(np.argmax(domain_probs, axis=-1)[0])
    domain_conf = float(domain_probs[0][domain_id])

    result = {
        "text": text,
        "domain": id2domain[domain_id],
        "domain_confidence": round(domain_conf, 4),
    }

    if domain_id == 0:  # camera
        camera_probs = softmax(camera_logits)
        pred_id = int(np.argmax(camera_probs, axis=-1)[0])
        confidence = float(camera_probs[0][pred_id])
        result["label"] = id2label[pred_id]
        result["confidence"] = round(confidence, 4)

    elif domain_id == 1:  # closure
        closure_probs = sigmoid(closure_logits)[0]
        actions = []
        for i in range(len(closure_probs)):
            if closure_probs[i] > 0.5:
                label = closure_id2label[i]
                parts = label.replace("closure.", "").split("_", 1)
                action = parts[0]
                target = parts[1] if len(parts) > 1 else ""
                actions.append(
                    {
                        "target": target,
                        "action": action,
                        "confidence": round(float(closure_probs[i]), 4),
                    }
                )
        result["actions"] = actions

    return result


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
