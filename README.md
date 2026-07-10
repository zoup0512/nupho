# nupho - Sophomore's Natural Language Understander

## Setup Environment

- Install uv. See: [UV Document](https://docs.astral.sh/uv/getting-started/installation/)
- Run: `uv sync` in Project root dir.

## Architecture

Multi-task dual-head model based on Chinese-MacBERT:

- **Domain head** (3-way softmax): `camera` / `closure` / `unknown`
- **Camera head** (15-way softmax): single-label classification for camera control intents
- **Closure head** (16-way sigmoid): multi-label classification for door/hood/trunk/fuel-cap open/close actions

One forward pass produces all three outputs. Loss is computed selectively based on domain label.

## Usage

- Train: `uv run .\src\train_classifier.py`
- Predict: `uv run .\src\predict.py --text "<user input>"`
- Export Onnx Model: `uv run .\src\export_onnx.py`
- Predict using Onnx Model: `uv run .\src\predict_onnx.py --text "<user input>"`

The onnx model is in `outputs/android` dir. Copy the file to android assets to implement model in android.

Needed files:

- `bert.onnx`
- `vocab.txt`
- `id2label.json` (camera label mapping)
- `id2domain.json` (domain mapping)
- `closure_id2label.json` (closure action mapping)
- `model_config.json`

## Data

- `data/dataset.jsonl`: Training data (with `domain` field).
- `data/label2id.json`: Camera label mapping.
- `data/domain2id.json`: Domain mapping.
- `data/closure_label2id.json`: Closure action mapping (8 targets x 2 actions).
- `assets/closure_operation.json`: Closure operation intent schema definition.
