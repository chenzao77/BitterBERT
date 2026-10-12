# BitterBERT

BitterBERT is an end-to-end chemical language model for molecular bitterness prediction. It fine-tunes **ChemBERTa-77M-MTR** directly on SMILES, using a balanced dataset with multi-taste negative sampling. Molecular representations and the binary classification head are learned jointly, without handcrafted molecular descriptors.

## Workflow

![Figure 2. Overall architecture and training workflow of BitterBERT.](Figure%202.png)

## Repository contents

| Path | Contents |
| --- | --- |
| `data/data.csv` | Training dataset: 3,200 molecules (1,600 bitter and 1,600 non-bitter). |
| `data/data_val.csv` | External test dataset: 262 molecules (131 bitter and 131 non-bitter). |
| `model/chemberta_fold_1/`–`model/chemberta_fold_5/` | Five fine-tuned models, each with its configuration, weights, and tokenizer files. |
| `model/train.py` | Five-fold training with training-fold SMILES augmentation and OOF predictions. |
| `model/predict.py` | Prediction by averaging the bitter-class probabilities of the five models. |

Both datasets contain `No`, `SMILES`, `Taste`, and `Label`. **Label 1 = bitter; label 0 = non-bitter.** Training negatives comprise 800 sweet, 480 miscellaneous (sour, salty, and umami), and 320 tasteless molecules (50:30:20).

The prediction script uses a default decision threshold of **0.5**. Five-model ensemble predictions on training molecules are not out-of-fold predictions. The training script saves genuine OOF predictions and its selected threshold, and writes retrained models to a separate directory.

The base model is [DeepChem/ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR). See its model card for the pretrained model's details.

## Citation

When using BitterBERT in research, please cite the associated manuscript: *BitterBERT: An End-to-End Chemical Language Model for Bitterness Prediction*. Publication details will be added when available.
