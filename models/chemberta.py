"""Pretrained ChemBERTa encoder with a readout head from models/readouts.py."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from models.readouts import build_readout


class Smiles(Dataset):
    """One molecule = its SMILES string plus its target values."""

    def __init__(self, frame, targets):
        self.smiles = frame.canonical_smiles.tolist()
        self.y = frame.reindex(columns=targets).to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        return {'smiles': self.smiles[i], 'y': torch.tensor(self.y[i])}


class ChemBERTaModel(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self._device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.checkpoint)
        self.encoder = AutoModel.from_pretrained(cfg.checkpoint).to(self._device)
        self.encoder.requires_grad_(not cfg.freeze_encoder)
        self.readout = build_readout(readout, self.encoder.config.hidden_size,
                                     n_targets).to(self._device)

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
        return DataLoader(Smiles(frame, targets), batch_size=batch_size,
                          shuffle=shuffle, collate_fn=self.collate, num_workers=0)

    def param_groups(self):
        groups = [{'params': list(self.readout.parameters()), 'lr': self.cfg.readout_lr}]
        if not self.cfg.freeze_encoder:
            groups.append({'params': list(self.encoder.parameters()),
                           'lr': self.cfg.encoder_lr})
        return groups

    def pool(self, hidden, mask):
        """[batch, tokens, dim] -> [batch, dim]."""
        if self.cfg.pooling == 'cls':
            return hidden[:, 0]
        mask = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, batch):
        ids = batch['input_ids'].to(self.device)
        mask = batch['attention_mask'].to(self.device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.readout(self.pool(hidden, mask))