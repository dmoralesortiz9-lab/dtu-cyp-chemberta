#!/usr/bin/env python3
"""How much molecular information does each pooled ChemBERTa summary carry?

A linear probe: pool the encoder output into one vector per molecule, then fit a ridge
regression from that vector to a known RDKit property and score it on held-out molecules.
A high R2 means the information is present AND linearly accessible; a negative one means
the vector is less useful than predicting the average.

Light enough for the login node (CPU, ~1 min); no training is involved.
    HF_HOME="$PWD/.hf_cache" python probe_pooling.py
"""
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from transformers import AutoModel, AutoTokenizer

CHECKPOINTS = ['DeepChem/ChemBERTa-77M-MLM', 'DeepChem/ChemBERTa-77M-MTR']
N_MOLECULES, N_FIT = 600, 400
PROPERTIES = {'MolWt': Descriptors.MolWt, 'MolLogP': Descriptors.MolLogP,
              'RingCount': Descriptors.RingCount}


def summaries(repo, smiles, batch_size=64):
    """Returns the CLS and masked-mean vectors for each molecule."""
    tokenizer = AutoTokenizer.from_pretrained(repo)
    encoder = AutoModel.from_pretrained(repo, add_pooling_layer=False).eval()
    cls, mean = [], []
    for i in range(0, len(smiles), batch_size):
        batch = tokenizer(smiles[i:i + batch_size], padding=True, truncation=True,
                          max_length=256, return_tensors='pt')
        with torch.no_grad():
            hidden = encoder(**batch).last_hidden_state
        mask = batch['attention_mask'].unsqueeze(-1).float()
        cls.append(hidden[:, 0])
        mean.append((hidden * mask).sum(1) / mask.sum(1))
    return torch.cat(cls).numpy(), torch.cat(mean).numpy()


def main():
    smiles = pd.read_csv('data/regression/train.csv').canonical_smiles.tolist()[:N_MOLECULES]
    molecules = [Chem.MolFromSmiles(s) for s in smiles]
    truth = {name: np.array([f(m) for m in molecules], dtype=float)
             for name, f in PROPERTIES.items()}
    print(f'{len(smiles)} molecules; fitting on {N_FIT}, scoring on {len(smiles) - N_FIT}\n')

    for repo in CHECKPOINTS:
        cls, mean = summaries(repo, smiles)
        options = {'cls': cls, 'mean': mean}
        # A weighted mix is lossy: it should land between the two, never above both.
        for weight in (0.25, 0.5, 0.75):
            options[f'mix {weight:.2f} cls'] = weight * cls + (1 - weight) * mean
        options['concat'] = np.concatenate([cls, mean], axis=1)

        print(f'=== {repo}   (R2 on held-out molecules)')
        print(f'{"pooling":16s}' + ''.join(f'{p:>12s}' for p in truth))
        for name, X in options.items():
            row = f'{name:16s}'
            for y in truth.values():
                model = Ridge(alpha=1.0).fit(X[:N_FIT], y[:N_FIT])
                row += f'{r2_score(y[N_FIT:], model.predict(X[N_FIT:])):12.3f}'
            print(row)
        print()


if __name__ == '__main__':
    main()
