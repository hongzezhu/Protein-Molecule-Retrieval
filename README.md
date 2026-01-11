Protein–Molecule Retrieval

Dual-tower retrieval between proteins and small molecules (SaProt protein encoder + SMILES Transformer).
This repository supports training, bidirectional retrieval evaluation, and an optional Gradio demo.

What this repo does

1.Protein → Molecule retrieval (virtual screening)
2.Molecule → Protein retrieval (target fishing)
3.Single pair scoring (protein–molecule compatibility)

Setup

4.Create an environment (example: Python 3.10)

conda create -n pmretrieval python=3.10 -y
conda activate pmretrieval

5.Install dependencies

# Install PyTorch first (choose the build that matches your CUDA)
# Example (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Core packages
pip install pytorch-lightning transformers pandas numpy pyyaml tqdm easydict lmdb requests

# Optional (demo)
pip install gradio

# Optional (molecule rendering in demo)
# Recommended via conda-forge:
conda install -c conda-forge rdkit -y

Data

This project can be built on the BindingDB dataset hosted on Hugging Face:
https://huggingface.co/datasets/vladak/bindingdb

The pipeline expects protein–SMILES pairs and writes them into LMDB for training/evaluation.

Option A: Download BindingDB and export to Parquet

6.Install Hugging Face datasets:

pip install datasets

7.Download and export (example code):

from datasets import load_dataset

ds = load_dataset("vladak/bindingdb")
# Inspect available splits and columns
print(ds)

# Export one split (adjust the split name if needed)
# Many datasets use "train" only; some may have multiple splits.
df = ds["train"].to_pandas()
df.to_parquet("bindingdb_train.parquet", index=False)

8.Place your parquet files into a directory, for example:

mkdir -p /path/to/bindingdb_parquets
mv bindingdb_train.parquet /path/to/bindingdb_parquets/

9.Build LMDB:

python dataset/prepare_bindingdb_pairs.py   --data_dir /path/to/bindingdb_parquets   --output LMDB/BindingDB   --split_ratios 0.9 0.05 0.05

Outputs:

LMDB/BindingDB/train
LMDB/BindingDB/valid
LMDB/BindingDB/test

Notes:
- Column names differ across BindingDB releases and preprocessors. The script attempts to map common names (SMILES, target sequence, UniProt ID, etc.).
- If your columns are non-standard, edit the mapping logic in dataset/prepare_bindingdb_pairs.py.

Training

Edit LMDB paths in dual_tower_baseline.yaml, then run:

python training.py -c dual_tower_baseline.yaml

Key fields in dual_tower_baseline.yaml:
- model.kwargs.protein_config_path
- dataset.train_lmdb, dataset.valid_lmdb, dataset.test_lmdb
- dataset.kwargs.max_protein_length, dataset.kwargs.max_molecular_length
- phased_training.enable (optional, two-stage training)

Evaluation

Set the checkpoint path and test LMDB path in evaluate_grouped.py, then run:

python evaluate_grouped.py

The script reports standard retrieval metrics in both directions (Hit@K, NDCG@K, MRR).

Random baseline:

python evaluate_random_baseline.py

Gradio demo (optional)

Set CHECKPOINT_PATH in app.py, then run:

python app.py

Default port: 7860.
