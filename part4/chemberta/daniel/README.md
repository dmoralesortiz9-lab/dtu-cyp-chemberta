# Part 4 - ChemBERTa-2: Daniel's work

Initial integration of ChemBERTa-2 into the starter pipeline, and the exploratory experiments that led to the selected configuration (MTR checkpoint, CLS pooling, fine-tuned encoder).

## Files

- chemberta_curves.pdf / .png: training and validation curves for the three configurations (initial MLM, tuned MLM, final MTR)
- resultados_chemberta.pdf: experiment log and observations (in Spanish)
- parte4_chemberta.tex: first draft of the report section
- plot_curves.py: script that generates the curve figures

## Note on the numbers

This folder documents the exploratory phase, run with a single seed per configuration. The figures and the log therefore show values that do not exactly match the report.

Every number in the report comes from the three-seed runs stored under part4/chemberta/miguel/results/, which are the reproducible ones. The model code itself (models/chemberta.py in the repository root) started here and was later extended by Miguel with the checkpoint selection, SMILES randomisation and the add_pooling_layer=False fix.
