"""Predict both the training set and the external validation set.

The five ChemBERTa folds are loaded once each. Predictions are then split back
into two independently sorted CSV files. Training-set ensemble predictions
are not out-of-fold predictions. Requires torch, numpy, pandas, transformers
(4.48.1), and safetensors. All paths are relative to this script's location.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Config:
    model_dir = Path(__file__).resolve().parent

    datasets = (
        {
            "source": "data",
            "input_path": model_dir.parent / "data" / "data.csv",
            "output_path": "data_predict_sorted.csv",
        },
        {
            "source": "data_val",
            "input_path": model_dir.parent / "data" / "data_val.csv",
            "output_path": "data_val_predict_sorted.csv",
        },
    )

    max_length = 256
    batch_size = 64
    threshold = 0.5


cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(dataset):
    """Load one dataset and standardize it to the prediction output schema."""
    df = pd.read_csv(dataset["input_path"])
    if "SMILES" not in df.columns:
        raise ValueError(f"{dataset['input_path']} does not contain a SMILES column")

    if df.empty or df["SMILES"].isna().any() or df["SMILES"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{dataset['input_path']} contains empty SMILES or no molecules")
    if "No" in df.columns:
        numbers = df["No"]
    elif "no" in df.columns:
        numbers = df["no"]
    else:
        numbers = pd.Series(range(1, len(df) + 1), index=df.index)

    true_labels = (
        df["Label"]
        if "Label" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    return pd.DataFrame(
        {
            "No": numbers,
            "SMILES": df["SMILES"].astype(str),
            "source": dataset["source"],
            "true_label": true_labels,
        }
    )


@torch.inference_mode()
def predict_one_fold(model_dir, smiles_list, tokenizer):
    """Predict the bitter-class probability using one fold."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, output_attentions=False, local_files_only=True
    ).to(device)
    model.eval()

    probs = []
    for i in range(0, len(smiles_list), cfg.batch_size):
        batch = smiles_list[i : i + cfg.batch_size]
        encoded = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        probs.append(softmax(logits, dim=-1)[:, 1].float().cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Optional CSV of new molecules with a SMILES column")
    parser.add_argument("--model-dir", type=Path, default=cfg.model_dir)
    parser.add_argument("--output-dir", type=Path, default=cfg.model_dir.parent / "results")
    parser.add_argument("--threshold", type=float, default=cfg.threshold)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    datasets = cfg.datasets if args.input is None else (
        {"source": "input", "input_path": args.input, "output_path": "predictions.csv"},
    )
    tables = [load_dataset(dataset) for dataset in datasets]
    combined = pd.concat(tables, ignore_index=True)
    smiles_list = combined["SMILES"].tolist()
    counts = ", ".join(
        f"{dataset['source']}: {len(table)}"
        for dataset, table in zip(datasets, tables)
    )
    print(f"Samples to predict: {len(combined)} ({counts})")

    fold_probs = []
    for fold in range(1, 6):
        model_dir = args.model_dir / f"chemberta_fold_{fold}"
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        print(f"Predicting with {model_dir} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        fold_probs.append(predict_one_fold(model_dir, smiles_list, tokenizer))

    ensemble_prob = np.mean(np.vstack(fold_probs), axis=0)
    combined["prob_bitter"] = ensemble_prob
    combined["predict"] = (ensemble_prob >= args.threshold).astype(int)

    output_columns = [
        "No",
        "SMILES",
        "source",
        "true_label",
        "predict",
        "prob_bitter",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        output = combined.loc[combined["source"] == dataset["source"]]
        output = output.sort_values("prob_bitter", ascending=False).reset_index(drop=True)
        output[output_columns].to_csv(
            args.output_dir / dataset["output_path"], index=False, encoding="utf-8-sig"
        )
        print(
            f"Saved {len(output)} predictions (sorted by prob_bitter) -> "
            f"{args.output_dir / dataset['output_path']}"
        )


if __name__ == "__main__":
    main()
