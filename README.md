# Protein-Molecule-Retrieval
Protein-Molecule Retrieval

Quick Start

cd Protein-Molecule-Retrieval

1) Prepare data (see below)

2) Train
python training.py -c dual_tower_baseline.yaml

3) Evaluate (requires a checkpoint)
python evaluate_grouped.py

4) Run demo (requires a checkpoint)
python app.py
