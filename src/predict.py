import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from train_classifier import MultiTaskBert


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def predict(text: str, model_dir: str = "outputs/bert", max_length: int = 64):
    import numpy as np

    model_path = Path(model_dir)

    with open(model_path / "model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config["model_name"]
    num_camera_labels = config["num_camera_labels"]
    num_closure_labels = config["num_closure_labels"]
    num_domains = config["num_domains"]

    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = MultiTaskBert(
        model_name,
        num_camera_labels=num_camera_labels,
        num_closure_labels=num_closure_labels,
        num_domains=num_domains,
    )
    model.load_state_dict(
        torch.load(model_path / "model.pt", map_location="cpu", weights_only=True)
    )

    id2label = load_json(model_path / "id2label.json")
    id2label = {int(k): v for k, v in id2label.items()}

    id2domain = load_json(model_path / "id2domain.json")
    id2domain = {int(k): v for k, v in id2domain.items()}

    closure_id2label = load_json(model_path / "closure_id2label.json")
    closure_id2label = {int(k): v for k, v in closure_id2label.items()}

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
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

    domain_logits = outputs["domain_logits"]
    camera_logits = outputs["camera_logits"]
    closure_logits = outputs["closure_logits"]

    domain_probs = F.softmax(domain_logits, dim=-1)
    domain_conf, domain_id = torch.max(domain_probs, dim=-1)
    domain_id = domain_id.item()
    domain_conf = domain_conf.item()

    result = {
        "text": text,
        "domain": id2domain[domain_id],
        "domain_confidence": round(domain_conf, 4),
        "load_time_sec": round(load_time, 4),
        "inference_time_sec": round(time.perf_counter() - infer_start, 4),
    }

    if domain_id == 0:  # camera
        camera_probs = F.softmax(camera_logits, dim=-1)
        camera_conf, camera_id = torch.max(camera_probs, dim=-1)
        result["label"] = id2label[camera_id.item()]
        result["confidence"] = round(camera_conf.item(), 4)

    elif domain_id == 1:  # closure
        closure_probs = torch.sigmoid(closure_logits).squeeze(0)
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
                        "confidence": round(closure_probs[i].item(), 4),
                    }
                )
        result["actions"] = actions

    return result


def main():
    import numpy as np  # noqa: F401

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
