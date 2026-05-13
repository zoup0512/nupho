# nupho

## Setup Environment

- Install uv. See: [UV Document](https://docs.astral.sh/uv/getting-started/installation/)
- Run: `uv sync` in Project root dir.

## Usage

- Train: `uv run .\src\train_classifier.py`
- Predict: `uv run .\src\predict.py --text "<user input>"`
- Export Onnx Model: `uv run .\src\export_onnx.py`
- Predict using Onnx Model: `uv run .\src\predict_onnx.py --text "<user input>"`

The onnx model is in `outputs/android` dir. Copy the file to android assets to implement model in android.

Needed files:

- `bert.onnx`
- `id2label.json`
- `vocab.txt`

## Data

- `data/dataset.jsonl`: Training data.
- `data/label2id.json`: Category.
