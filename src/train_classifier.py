import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class MultiTaskBert(nn.Module):
    def __init__(self, model_name, num_camera_labels, num_closure_labels, num_domains,
                 closure_pos_weight=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden, num_domains)
        self.camera_head = nn.Linear(hidden, num_camera_labels)
        self.closure_head = nn.Linear(hidden, num_closure_labels)
        self.closure_pos_weight = closure_pos_weight

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        domain_labels=None,
        camera_labels=None,
        closure_labels=None,
    ):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.bert(**kwargs)
        pooled = self.dropout(outputs.last_hidden_state[:, 0])

        domain_logits = self.domain_head(pooled)
        camera_logits = self.camera_head(pooled)
        closure_logits = self.closure_head(pooled)

        result = {
            "domain_logits": domain_logits,
            "camera_logits": camera_logits,
            "closure_logits": closure_logits,
        }

        if domain_labels is not None:
            loss_domain = F.cross_entropy(domain_logits, domain_labels)

            loss_camera = torch.tensor(0.0, device=domain_logits.device)
            camera_mask = domain_labels == 0
            if camera_mask.any():
                loss_camera = F.cross_entropy(
                    camera_logits[camera_mask], camera_labels[camera_mask]
                )

            loss_closure = torch.tensor(0.0, device=domain_logits.device)
            closure_mask = domain_labels == 1
            if closure_mask.any():
                pw = self.closure_pos_weight
                if pw is not None and pw.device != closure_logits.device:
                    pw = pw.to(closure_logits.device)
                loss_closure = F.binary_cross_entropy_with_logits(
                    closure_logits[closure_mask],
                    closure_labels[closure_mask].float(),
                    pos_weight=pw,
                )

            result["loss"] = loss_domain + loss_camera + loss_closure

        return result


class MultiTaskDataCollator:
    def __init__(self, tokenizer, num_closure_labels, closure_label2id, max_length=64):
        self.tokenizer = tokenizer
        self.num_closure_labels = num_closure_labels
        self.closure_label2id = closure_label2id
        self.max_length = max_length

    def __call__(self, features):
        texts = [f["text"] for f in features]
        encoded = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]

        batch["domain_labels"] = torch.tensor([f["domain_id"] for f in features])

        camera_labels = []
        closure_labels = []
        for f in features:
            camera_labels.append(f.get("camera_label", -1))
            vec = [0.0] * self.num_closure_labels
            for a in (f.get("closure_actions") or []):
                key = f"closure.{a['action']}_{a['target']}"
                if key in self.closure_label2id:
                    vec[self.closure_label2id[key]] = 1.0
            closure_labels.append(vec)

        batch["camera_labels"] = torch.tensor(camera_labels, dtype=torch.long)
        batch["closure_labels"] = torch.tensor(closure_labels, dtype=torch.float)

        return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.jsonl")
    parser.add_argument("--camera_label2id", default="data/camera_label2id.json")
    parser.add_argument("--domain2id", default="data/domain2id.json")
    parser.add_argument("--closure_label2id", default="data/closure_label2id.json")
    parser.add_argument("--model_name", default="hfl/chinese-macbert-base")
    parser.add_argument("--output_dir", default="outputs/bert")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()

    data_path = Path(args.data)
    camera_label_path = Path(args.camera_label2id)
    domain_path = Path(args.domain2id)
    closure_label_path = Path(args.closure_label2id)

    rows = load_jsonl(data_path)

    with open(camera_label_path, "r", encoding="utf-8") as f:
        camera_label2id = json.load(f)
    with open(domain_path, "r", encoding="utf-8") as f:
        domain2id = json.load(f)
    with open(closure_label_path, "r", encoding="utf-8") as f:
        closure_label2id = json.load(f)

    camera_id2label = {v: k for k, v in camera_label2id.items()}
    id2domain = {v: k for k, v in domain2id.items()}

    processed = []
    for row in rows:
        item = {"text": row["text"], "domain_id": domain2id[row["domain"]]}
        if row["domain"] == "camera":
            item["camera_label"] = camera_label2id[row["label"]]
        else:
            item["camera_label"] = -1
        if row["domain"] == "closure":
            item["closure_actions"] = row.get("actions", [])
        processed.append(item)

    train_proc, dev_proc = train_test_split(
        processed,
        test_size=0.2,
        random_state=42,
        stratify=[p["domain_id"] for p in processed],
    )

    train_ds = Dataset.from_list(train_proc)
    dev_ds = Dataset.from_list(dev_proc)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    closure_pos_counts = [0] * len(closure_label2id)
    closure_total = 0
    for p in train_proc:
        if p["domain_id"] == 1:
            closure_total += 1
            for a in (p.get("closure_actions") or []):
                key = f"closure.{a['action']}_{a['target']}"
                if key in closure_label2id:
                    closure_pos_counts[closure_label2id[key]] += 1
    closure_pos_weight = torch.tensor([
        closure_total / max(c, 1) for c in closure_pos_counts
    ], dtype=torch.float)
    print(f"Closure pos_weight: {closure_pos_weight.tolist()}")

    model = MultiTaskBert(
        args.model_name,
        num_camera_labels=len(camera_label2id),
        num_closure_labels=len(closure_label2id),
        num_domains=len(domain2id),
        closure_pos_weight=closure_pos_weight,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        modules_to_save=["domain_head", "camera_head", "closure_head"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = MultiTaskDataCollator(
        tokenizer, num_closure_labels=len(closure_label2id),
        closure_label2id=closure_label2id, max_length=args.max_length
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # Trainer returns predictions as tuple of arrays
        # (domain_logits, camera_logits, closure_logits)
        if isinstance(predictions, tuple):
            domain_logits = predictions[0]
            camera_logits = predictions[1]
            closure_logits = predictions[2]
        else:
            domain_logits = predictions["domain_logits"]
            camera_logits = predictions["camera_logits"]
            closure_logits = predictions["closure_logits"]

        # Trainer returns labels as tuple matching label_names order
        # (domain_labels, camera_labels, closure_labels)
        if isinstance(labels, tuple):
            domain_labels = labels[0]
            camera_labels_arr = labels[1]
            closure_labels_arr = labels[2]
        else:
            domain_labels = labels["domain_labels"]
            camera_labels_arr = labels["camera_labels"]
            closure_labels_arr = labels["closure_labels"]

        domain_preds = np.argmax(domain_logits, axis=-1)

        domain_acc = accuracy_score(domain_labels, domain_preds)
        domain_f1 = f1_score(domain_labels, domain_preds, average="macro")

        result = {
            "domain_accuracy": domain_acc,
            "domain_macro_f1": domain_f1,
        }

        camera_mask = domain_labels == 0
        if camera_mask.any():
            cam_preds = np.argmax(camera_logits[camera_mask], axis=-1)
            cam_labels = camera_labels_arr[camera_mask]
            valid = cam_labels >= 0
            if valid.any():
                result["camera_accuracy"] = accuracy_score(
                    cam_labels[valid], cam_preds[valid]
                )
                result["camera_macro_f1"] = f1_score(
                    cam_labels[valid], cam_preds[valid], average="macro"
                )

        closure_mask = domain_labels == 1
        if closure_mask.any():
            clo_preds = (closure_logits[closure_mask] > 0.5).astype(int)
            clo_labels = closure_labels_arr[closure_mask].astype(int)
            result["closure_f1"] = f1_score(
                clo_labels, clo_preds, average="macro", zero_division=0
            )

        return result

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
        metric_for_best_model="domain_macro_f1",
        greater_is_better=True,
        fp16=use_cuda,
        report_to="none",
        remove_unused_columns=False,
        label_names=["domain_labels", "camera_labels", "closure_labels"],
    )

    class TorchTrainer(Trainer):
        def _save(self, output_dir=None, state_dict=None):
            if output_dir is None:
                output_dir = self.args.output_dir
            import os
            os.makedirs(output_dir, exist_ok=True)
            if state_dict is None:
                state_dict = self.model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.pt"))

    trainer = TorchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    torch.save(merged_model.state_dict(), output_dir / "model.pt")
    tokenizer.save_pretrained(output_dir)

    with open(output_dir / "camera_label2id.json", "w", encoding="utf-8") as f:
        json.dump(camera_label2id, f, ensure_ascii=False, indent=2)
    with open(output_dir / "camera_id2label.json", "w", encoding="utf-8") as f:
        json.dump(camera_id2label, f, ensure_ascii=False, indent=2)
    with open(output_dir / "domain2id.json", "w", encoding="utf-8") as f:
        json.dump(domain2id, f, ensure_ascii=False, indent=2)
    with open(output_dir / "id2domain.json", "w", encoding="utf-8") as f:
        json.dump(id2domain, f, ensure_ascii=False, indent=2)
    with open(output_dir / "closure_label2id.json", "w", encoding="utf-8") as f:
        json.dump(closure_label2id, f, ensure_ascii=False, indent=2)

    closure_id2label = {v: k for k, v in closure_label2id.items()}
    with open(output_dir / "closure_id2label.json", "w", encoding="utf-8") as f:
        json.dump(closure_id2label, f, ensure_ascii=False, indent=2)

    config = {
        "model_name": args.model_name,
        "num_camera_labels": len(camera_label2id),
        "num_closure_labels": len(closure_label2id),
        "num_domains": len(domain2id),
        "max_length": args.max_length,
    }
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
