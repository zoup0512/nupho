import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class MultiTaskBert(nn.Module):
    def __init__(self, model_name, num_camera_labels, num_closure_labels, num_domains):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden, num_domains)
        self.camera_head = nn.Linear(hidden, num_camera_labels)
        self.closure_head = nn.Linear(hidden, num_closure_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.bert(**kwargs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        return {
            "domain_logits": self.domain_head(pooled),
            "camera_logits": self.camera_head(pooled),
            "closure_logits": self.closure_head(pooled),
        }


class OnnxWrapper(nn.Module):
    def __init__(self, model: MultiTaskBert):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids)
        return out["domain_logits"], out["camera_logits"], out["closure_logits"]


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Skip, not found: {src}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/bert")
    parser.add_argument("--output_dir", default="outputs/android")
    parser.add_argument("--onnx_name", default="bert.onnx")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / args.onnx_name

    with open(model_dir / "model_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    model_name = config["model_name"]
    num_camera_labels = config["num_camera_labels"]
    num_closure_labels = config["num_closure_labels"]
    num_domains = config["num_domains"]

    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    model = MultiTaskBert(
        model_name,
        num_camera_labels=num_camera_labels,
        num_closure_labels=num_closure_labels,
        num_domains=num_domains,
    )
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()
    model.cpu()

    wrapper = OnnxWrapper(model)
    wrapper.eval()
    wrapper.cpu()

    dummy_text = "看一下车头"

    encoded = tokenizer(
        dummy_text,
        max_length=args.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if "token_type_ids" in encoded:
        token_type_ids = encoded["token_type_ids"]
    else:
        token_type_ids = torch.zeros_like(input_ids)

    print("Exporting ONNX model...")

    torch.onnx.export(
        wrapper,
        args=(input_ids, attention_mask, token_type_ids),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["domain_logits", "camera_logits", "closure_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "domain_logits": {0: "batch_size"},
            "camera_logits": {0: "batch_size"},
            "closure_logits": {0: "batch_size"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )

    print(f"ONNX model saved to: {onnx_path}")

    vocab_path = output_dir / "vocab.txt"

    if (model_dir / "vocab.txt").exists():
        copy_if_exists(model_dir / "vocab.txt", vocab_path)
    else:
        vocab = tokenizer.get_vocab()
        id_to_token = sorted(vocab.items(), key=lambda x: x[1])

        with open(vocab_path, "w", encoding="utf-8") as f:
            for token, _ in id_to_token:
                f.write(token + "\n")

    print(f"Generated: {vocab_path}")
    copy_if_exists(model_dir / "tokenizer.json", output_dir / "tokenizer.json")
    copy_if_exists(
        model_dir / "tokenizer_config.json", output_dir / "tokenizer_config.json"
    )
    copy_if_exists(
        model_dir / "special_tokens_map.json", output_dir / "special_tokens_map.json"
    )
    copy_if_exists(model_dir / "id2label.json", output_dir / "id2label.json")
    copy_if_exists(model_dir / "label2id.json", output_dir / "label2id.json")
    copy_if_exists(model_dir / "id2domain.json", output_dir / "id2domain.json")
    copy_if_exists(model_dir / "domain2id.json", output_dir / "domain2id.json")
    copy_if_exists(
        model_dir / "closure_id2label.json", output_dir / "closure_id2label.json"
    )
    copy_if_exists(
        model_dir / "closure_label2id.json", output_dir / "closure_label2id.json"
    )
    copy_if_exists(model_dir / "model_config.json", output_dir / "model_config.json")

    print("Done.")
    print()
    print("Android assets should include:")
    print(f"- {onnx_path}")
    print(f"- {output_dir / 'vocab.txt'}")
    print(f"- {output_dir / 'id2label.json'}")
    print(f"- {output_dir / 'id2domain.json'}")
    print(f"- {output_dir / 'closure_id2label.json'}")
    print(f"- {output_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()
