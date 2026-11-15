"""Phase A, step 1: load ChemBERTa-2 standalone and push one real SMILES through it.

Shows: SMILES -> tokens -> (n_tokens, 384) table -> pooled (384,) vector.
"""
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
smiles = pd.read_csv("data/regression/train.csv", nrows=1)["SMILES"].iloc[0]
print(f"device={device}  SMILES={smiles}\n")

for repo in ["DeepChem/ChemBERTa-77M-MLM", "DeepChem/ChemBERTa-77M-MTR"]:
    tok = AutoTokenizer.from_pretrained(repo)
    # add_pooling_layer=False: we do our own pooling, the built-in pooler is not pretrained.
    model = AutoModel.from_pretrained(repo, add_pooling_layer=False).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    batch = tok([smiles], return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state  # (1, n_tokens, 384)

    mask = batch["attention_mask"].unsqueeze(-1).float()  # 1 = real token, 0 = padding
    mean_vec = (hidden * mask).sum(1) / mask.sum(1)        # (1, 384) average over tokens
    cls_vec = hidden[:, 0]                                  # (1, 384) first (<s>) token

    print(f"=== {repo}  ({n_params/1e6:.2f} M weights)")
    print("tokens :", tok.convert_ids_to_tokens(batch["input_ids"][0]))
    print("table  :", tuple(hidden.shape), "-> (molecules, tokens, dims)")
    print("mean   :", tuple(mean_vec.shape), "first 5:", mean_vec[0, :5].cpu().numpy().round(3))
    print("cls    :", tuple(cls_vec.shape), "first 5:", cls_vec[0, :5].cpu().numpy().round(3))
    print("max len:", tok.model_max_length, "| vocab:", tok.vocab_size, "\n")
