import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request, send_from_directory
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import MultiTaskBert

app = Flask(__name__, static_folder=None)

MODEL_DIR = Path("outputs/bert")
MAX_LENGTH = 64
TEST_SET = Path("data/test_set.jsonl")
FEEDBACK_FILE = Path("data/feedback.jsonl")
RETRAIN_DATA = Path("data/retrain_data.jsonl")

_model = None
_tokenizer = None
_id2label = {}
_id2domain = {}
_closure_id2label = {}


def load_model():
    global _model, _tokenizer, _id2label, _id2domain, _closure_id2label

    with open(MODEL_DIR / "model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    _model = MultiTaskBert(
        config["model_name"],
        num_camera_labels=config["num_camera_labels"],
        num_closure_labels=config["num_closure_labels"],
        num_domains=config["num_domains"],
    )
    _model.load_state_dict(
        torch.load(MODEL_DIR / "model.pt", map_location="cpu", weights_only=True),
        strict=False,
    )
    _model.eval()

    with open(MODEL_DIR / "camera_id2label.json", "r", encoding="utf-8") as f:
        _id2label = {int(k): v for k, v in json.load(f).items()}
    with open(MODEL_DIR / "id2domain.json", "r", encoding="utf-8") as f:
        _id2domain = {int(k): v for k, v in json.load(f).items()}
    with open(MODEL_DIR / "closure_id2label.json", "r", encoding="utf-8") as f:
        _closure_id2label = {int(k): v for k, v in json.load(f).items()}


def run_predict(text):
    inputs = _tokenizer(
        text, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )
    with torch.no_grad():
        out = _model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
        )

    domain_probs = F.softmax(out["domain_logits"], dim=-1)
    domain_conf, domain_id = torch.max(domain_probs, dim=-1)
    domain_id = domain_id.item()
    domain_conf = round(domain_conf.item(), 4)

    result = {"domain": _id2domain[domain_id], "domain_confidence": domain_conf}

    if domain_id == 0:
        camera_probs = F.softmax(out["camera_logits"], dim=-1)
        cam_conf, cam_id = torch.max(camera_probs, dim=-1)
        result["label"] = _id2label[cam_id.item()]
        result["confidence"] = round(cam_conf.item(), 4)

    elif domain_id == 1:
        closure_probs = torch.sigmoid(out["closure_logits"]).squeeze(0)
        raw_actions = []
        for i in range(len(closure_probs)):
            if closure_probs[i] > 0.5:
                label = _closure_id2label[i]
                parts = label.replace("closure.", "").split("_", 1)
                action = parts[0]
                target = parts[1] if len(parts) > 1 else ""
                raw_actions.append({
                    "target": target,
                    "action": action,
                    "confidence": round(closure_probs[i].item(), 4),
                })
        best_by_target = {}
        for a in raw_actions:
            t = a["target"]
            if t not in best_by_target or a["confidence"] > best_by_target[t]["confidence"]:
                best_by_target[t] = a
        result["actions"] = list(best_by_target.values())

    return result


def load_test_cases():
    with open(TEST_SET, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent / "templates", "index.html")


@app.route("/api/test_cases")
def get_test_cases():
    cases = load_test_cases()
    return jsonify(cases)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    result = run_predict(text)
    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    entry = {
        "text": data["text"],
        "expected_domain": data["expected_domain"],
        "expected_label": data.get("expected_label"),
        "expected_actions": data.get("expected_actions"),
        "predicted": data["predicted"],
        "is_correct": data["is_correct"],
        "corrected_domain": data.get("corrected_domain"),
        "corrected_label": data.get("corrected_label"),
        "corrected_actions": data.get("corrected_actions"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"status": "ok"})


@app.route("/api/feedback/stats")
def feedback_stats():
    if not FEEDBACK_FILE.exists():
        return jsonify({"total": 0, "correct": 0, "incorrect": 0})
    correct = 0
    incorrect = 0
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["is_correct"]:
                correct += 1
            else:
                incorrect += 1
    return jsonify({"total": correct + incorrect, "correct": correct, "incorrect": incorrect})


@app.route("/api/retrain", methods=["POST"])
def retrain():
    if not FEEDBACK_FILE.exists():
        return jsonify({"status": "error", "message": "No feedback data"})

    feedback_entries = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            feedback_entries.append(json.loads(line))

    retrain_samples = []
    for fb in feedback_entries:
        if fb["is_correct"]:
            continue
        domain = fb.get("corrected_domain") or fb["expected_domain"]
        sample = {"text": fb["text"], "domain": domain}
        if domain == "camera":
            sample["label"] = fb.get("corrected_label") or fb.get("expected_label")
        elif domain == "closure":
            sample["actions"] = fb.get("corrected_actions") or fb.get("expected_actions")
        retrain_samples.append(sample)

    if not retrain_samples:
        return jsonify({"status": "error", "message": "No incorrect feedback to learn from"})

    with open(RETRAIN_DATA, "a", encoding="utf-8") as f:
        for s in retrain_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    return jsonify({
        "status": "ok",
        "message": f"Added {len(retrain_samples)} correction samples to {RETRAIN_DATA}. "
                   f"Run training: uv run python src/train_classifier.py --data data/dataset.jsonl",
        "samples_added": len(retrain_samples),
    })


@app.route("/api/retrain_data/clear", methods=["POST"])
def clear_retrain_data():
    if RETRAIN_DATA.exists():
        RETRAIN_DATA.unlink()
    if FEEDBACK_FILE.exists():
        FEEDBACK_FILE.unlink()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Model loaded.")
    app.run(host="0.0.0.0", port=5000, debug=False)
