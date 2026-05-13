import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/camera_dataset.jsonl")
    parser.add_argument("--label2id", default="data/label2id.json")
    parser.add_argument("--model_name", default="hfl/chinese-macbert-base")
    parser.add_argument("--output_dir", default="outputs/camera-bert")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    data_path = Path(args.data)
    label_path = Path(args.label2id)

    rows = load_jsonl(data_path)

    with open(label_path, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    id2label = {v: k for k, v in label2id.items()}

    texts = [x["text"] for x in rows]
    labels = [label2id[x["label"]] for x in rows]

    train_texts, dev_texts, train_labels, dev_labels = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    train_ds = Dataset.from_dict({
        "text": train_texts,
        "label": train_labels,
    })

    dev_ds = Dataset.from_dict({
        "text": dev_texts,
        "label": dev_labels,
    })

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    dev_ds = dev_ds.map(tokenize, batched=True)

    train_ds = train_ds.remove_columns(["text"])
    dev_ds = dev_ds.remove_columns(["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=use_cuda,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(output_dir / "label2id.json", "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(id2label, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
