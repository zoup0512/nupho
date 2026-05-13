import argparse
import json
import shutil
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BertClassifierOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")
    else:
        print(f"Skip, not found: {src}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/camera-bert")
    parser.add_argument("--output_dir", default="outputs/android")
    parser.add_argument("--onnx_name", default="camera_bert.onnx")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / args.onnx_name

    print(f"Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    model.eval()
    model.cpu()

    wrapper = BertClassifierOnnxWrapper(model)
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
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "token_type_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
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

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        android_config = {
            "max_length": args.max_length,
            "num_labels": config.get("num_labels"),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
        }

        with open(output_dir / "camera_model_config.json", "w", encoding="utf-8") as f:
            json.dump(android_config, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_dir / 'camera_model_config.json'}")

    print("Done.")
    print()
    print("Android assets should include:")
    print(f"- {onnx_path}")
    print(f"- {output_dir / 'vocab.txt'}")
    print(f"- {output_dir / 'id2label.json'}")


if __name__ == "__main__":
    main()
