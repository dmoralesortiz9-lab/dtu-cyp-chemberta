"""Quick check: does ChemBERTa download and run?"""
import torch
from transformers import AutoTokenizer, AutoModel

NAME = "DeepChem/ChemBERTa-77M-MLM"

tokenizer = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME)
print("Model loaded successfully")

# Two example molecules as SMILES strings: ethanol and aspirin
smiles = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]

batch = tokenizer(smiles, padding=True, return_tensors="pt")
print("Tokens per molecule:", batch["input_ids"].shape)

with torch.no_grad():
    output = model(**batch).last_hidden_state

print("Output:", output.shape)
print("Embedding size:", output.shape[-1])