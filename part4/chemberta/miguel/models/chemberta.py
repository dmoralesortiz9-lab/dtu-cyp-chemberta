"""Pretrained ChemBERTa encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from transformers import AutoConfig, AutoModel, AutoTokenizer
from models.readouts import build_readout


class Smiles(Dataset):
    """One molecule = its SMILES string plus its target values.

    With augment=True every access writes the molecule a different way (SMILES
    enumeration): the atoms are renumbered at random and the string is written
    non-canonically. The molecule is identical, the text the model sees is not.
    """

    def __init__(self, frame, targets, augment=False, seed=0):
        self.smiles = frame.canonical_smiles.tolist()
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.mols = [Chem.MolFromSmiles(s) for s in self.smiles] if augment else None

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        smiles = self.smiles[i]
        if self.augment:
            mol = self.mols[i]
            order = self.rng.permutation(mol.GetNumAtoms()).tolist()
            smiles = Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)
        return {'smiles': smiles, 'y': torch.tensor(self.y[i])}


class ChemBERTaModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        config = AutoConfig.from_pretrained(cfg.checkpoint)
        # add_pooling_layer=False: the pooler is absent from these checkpoints, so it would
        # be randomly initialised, and forward() uses last_hidden_state anyway.
        if cfg.pretrained:
            self.encoder = AutoModel.from_pretrained(cfg.checkpoint, add_pooling_layer=False)
        else:
            # Ablation: same architecture, random weights, to measure what pretraining adds.
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        self.encoder = self.encoder.to(self._device)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        # 'concat' feeds the readout both summaries, so its input is twice as wide.
        in_dim = config.hidden_size * (2 if cfg.pooling == 'concat' else 1)
        self.readout = build_readout(readout, in_dim, n_targets).to(self._device)

    @property
    def device(self):
        return self._device

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.freeze_encoder:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def collate(self, items):
        """Tokenize a list of molecules into one padded batch."""
        batch = self.tokenizer([it['smiles'] for it in items], padding=True,
                               truncation=True, max_length=self.cfg.max_length,
                               return_tensors='pt')
        batch['y'] = torch.stack([it['y'] for it in items])
        return batch

    def loader(self, frame, targets, batch_size, shuffle=False):
        # Only the training loader shuffles, so it is also the only one to augment:
        # validation and test must stay deterministic.
        data = Smiles(frame, targets, augment=shuffle and self.cfg.augment,
                      seed=torch.initial_seed() % 2**32)
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=self.collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()),
                           'lr': self.cfg.encoder_lr})
        return groups

    def pool(self, hidden, mask):
        """[batch, tokens, dim] -> [batch, dim], or [batch, 2*dim] for 'concat'."""
        if self.cfg.pooling == 'cls':
            return hidden[:, 0]
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.cfg.pooling == 'mean':
            return mean
        if self.cfg.pooling == 'concat':
            return torch.cat([hidden[:, 0], mean], dim=1)
        raise ValueError(f'Unknown pooling {self.cfg.pooling!r}')

    def forward(self, batch):
        ids = batch['input_ids'].to(self.device)
        mask = batch['attention_mask'].to(self.device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.readout(self.pool(hidden, mask))
