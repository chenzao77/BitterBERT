"""Train five BitterBERT folds and save genuine out-of-fold predictions.

Run from any directory. Requires torch, transformers (4.48.1), datasets,
accelerate, pandas, numpy, scikit-learn, rdkit, and safetensors.
Retrained models are saved separately from the published weights.
"""

import argparse
import json
import gc
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from datasets import Dataset

from rdkit import Chem
from rdkit import rdBase

# ==========================================
# 1. Configuration
# ==========================================
class Config:
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    data_path = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    output_dir = Path(__file__).resolve().parent / "retrained"
    max_length = 256
    batch_size = 32
    epochs = 30
    learning_rate = 2e-5
    seed = 42
    num_folds = 5

    enable_smiles_augmentation = True

cfg = Config()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rdBase.SeedRandomNumberGenerator(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def randomize_smiles(smi):
    molecule = Chem.MolFromSmiles(smi)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return Chem.MolToSmiles(
        molecule, canonical=False, doRandom=True, isomericSmiles=True
    )

def augment_dataframe(df, num_aug=4):
    aug_rows = []
    for _, row in df.iterrows():
        aug_rows.append(row)
        for _ in range(num_aug):
            new_row = row.copy()
            new_row['SMILES'] = randomize_smiles(row['SMILES'])
            aug_rows.append(new_row)
    return pd.DataFrame(aug_rows).drop_duplicates(subset=['SMILES']).reset_index(drop=True)

# ==========================================
# 2. Validation metrics for epoch monitoring
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    probs = softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(labels, probs),
        "f1": f1_score(labels, preds)
    }

# ==========================================
# 3. Stratified five-fold cross-validation
# ==========================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=cfg.data_path)
    parser.add_argument("--output-dir", type=Path, default=cfg.output_dir)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    published_dir = Path(__file__).resolve().parent
    published_folds = [published_dir / f"chemberta_fold_{i}" for i in range(1, 6)]
    if output_dir == published_dir or any(
        output_dir == fold or fold in output_dir.parents for fold in published_folds
    ):
        raise ValueError("Choose an output directory outside the published model folders.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    print("Starting BitterBERT five-fold training...")

    # Preserve row order for OOF assignment; reject incomplete data.
    df = pd.read_csv(args.data).reset_index(drop=True)
    if df.empty or df[['SMILES', 'Label']].isna().any().any():
        raise ValueError("Training data must contain nonempty SMILES and Label values.")
    if df['SMILES'].astype(str).str.strip().eq('').any():
        raise ValueError("Training data contains blank SMILES.")
    if not df['Label'].isin([0, 1]).all() or df['Label'].nunique() != 2:
        raise ValueError("Training labels must contain both 0 (non-bitter) and 1 (bitter).")
    if df['SMILES'].duplicated().any():
        raise ValueError("Remove duplicate molecules before cross-validation.")
    if df['Label'].value_counts().min() < cfg.num_folds:
        raise ValueError("Each class must contain at least five molecules.")
    df.rename(columns={'Label': 'label'}, inplace=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    y_true = df['label'].values

    oof_probs = np.zeros(len(df))
    oof_folds = np.zeros(len(df), dtype=int)
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, y_true)):
        print("\n" + "="*50)
        print(f"Training fold {fold + 1} / {cfg.num_folds}")
        print("="*50)

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        if cfg.enable_smiles_augmentation:
            train_df = augment_dataframe(train_df, num_aug=4)

        # Only SMILES and labels are used as model inputs.
        train_dataset = Dataset.from_pandas(train_df[['SMILES', 'label']], preserve_index=False)
        val_dataset = Dataset.from_pandas(val_df[['SMILES', 'label']], preserve_index=False)

        def tokenize_function(examples):
            return tokenizer(examples['SMILES'], padding='max_length', truncation=True, max_length=cfg.max_length)

        train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['SMILES'])
        val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['SMILES'])

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2, output_attentions=False,
            id2label={0: "non-bitter", 1: "bitter"},
            label2id={"non-bitter": 0, "bitter": 1},
        )

        fold_dir = output_dir / f'chemberta_fold_{fold+1}'
        has_cuda = torch.cuda.is_available()
        supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
        supports_tf32 = has_cuda and torch.cuda.get_device_capability()[0] >= 8

        training_args = TrainingArguments(
            output_dir=str(fold_dir),
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            num_train_epochs=cfg.epochs,
            weight_decay=0.05,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            label_smoothing_factor=0.1,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            logging_steps=10,
            report_to="none",
            seed=cfg.seed,
            data_seed=cfg.seed,
            dataloader_num_workers=0,
            dataloader_pin_memory=has_cuda,
            bf16=supports_bf16,
            tf32=supports_tf32,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

        trainer.train()

        trainer.save_model(str(fold_dir))
        tokenizer.save_pretrained(str(fold_dir))

        predictions = trainer.predict(val_dataset)
        logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
        fold_probs = softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

        oof_probs[val_idx] = fold_probs
        oof_folds[val_idx] = fold + 1

        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ==========================================
    # 4. Select a decision threshold using pooled OOF F1.
    # ==========================================
    print("\n" + "="*80)
    print("Training complete. Selecting the OOF F1 threshold...")

    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.30, 0.85, 0.01):
        preds = (oof_probs >= th).astype(int)

        current_f1 = f1_score(y_true, preds)

        if current_f1 > best_f1:
            best_f1, best_th = current_f1, th

    final_preds = (oof_probs >= best_th).astype(int)

    auc_val = roc_auc_score(y_true, oof_probs)
    prc_val = average_precision_score(y_true, oof_probs)
    acc_val = accuracy_score(y_true, final_preds)
    pre_val = precision_score(y_true, final_preds, zero_division=0)
    rec_val = recall_score(y_true, final_preds, zero_division=0)

    oof = df.rename(columns={'label': 'Label'}).copy()
    oof['fold'] = oof_folds
    oof['prob_bitter'] = oof_probs
    oof['predict'] = final_preds
    oof.to_csv(output_dir / 'oof_predictions.csv', index=False)
    metrics = {
        'threshold': float(best_th), 'auc': float(auc_val),
        'f1': float(best_f1), 'auprc': float(prc_val),
        'accuracy': float(acc_val), 'precision': float(pre_val),
        'recall': float(rec_val),
    }
    (output_dir / 'oof_metrics.json').write_text(
        json.dumps(metrics, indent=2) + '\n', encoding='utf-8'
    )

    print("="*80)
    print(f"Best OOF F1 decision threshold: {best_th:.2f}")
    print(f"AUC: {auc_val:.4f} | F1: {best_f1:.4f} | PRC: {prc_val:.4f} | Acc: {acc_val:.4f} | Pre: {pre_val:.4f} | Rec: {rec_val:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
