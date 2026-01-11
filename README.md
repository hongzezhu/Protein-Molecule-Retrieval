# Protein–Molecule Retrieval

Dual-tower retrieval between proteins and small molecules (SaProt protein encoder + SMILES Transformer).
This repository supports training, bidirectional retrieval evaluation, and an optional Gradio demo.

What this repo does

1.Protein → Molecule retrieval (virtual screening)
2.Molecule → Protein retrieval (target fishing)
3.Single pair scoring (protein–molecule compatibility)

Setup

# 1.Create an environment (example: Python 3.10)
```bash
conda create -n pmretrieval python=3.10 -y
conda activate pmretrieval
```
# 2.Install dependencies

Install PyTorch first (choose the build that matches your CUDA)
Example (CUDA 12.1):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
Core packages
```bash
pip install pytorch-lightning transformers pandas numpy pyyaml tqdm easydict lmdb requests
```
Optional (demo)
```bash
pip install gradio
```
# 3.Data
This project can be built on the BindingDB dataset hosted on Hugging Face:
https://huggingface.co/datasets/vladak/bindingdb

The pipeline expects protein–SMILES pairs and writes them into LMDB for training/evaluation.

 Download BindingDB and export to Parquet

Build LMDB:
```bash
python dataset/prepare_bindingdb_pairs.py   --data_dir /path/to/bindingdb_parquets   --output LMDB/BindingDB   --split_ratios 0.9 0.05 0.05
```
Outputs:

LMDB/BindingDB/train
LMDB/BindingDB/valid
LMDB/BindingDB/test

Notes:
- Column names differ across BindingDB releases and preprocessors. The script attempts to map common names (SMILES, target sequence, UniProt ID, etc.).
- If your columns are non-standard, edit the mapping logic in dataset/prepare_bindingdb_pairs.py.

# 4.Training

Edit LMDB paths in dual_tower_baseline.yaml, then run:
```bash
python training.py -c dual_tower_baseline.yaml
```
Key fields in dual_tower_baseline.yaml:
- model.kwargs.protein_config_path
- dataset.train_lmdb, dataset.valid_lmdb, dataset.test_lmdb
- dataset.kwargs.max_protein_length, dataset.kwargs.max_molecular_length
- phased_training.enable (optional, two-stage training)

# 5.Evaluation

Set the checkpoint path and test LMDB path in evaluate_grouped.py, then run:
```bash
python evaluate_grouped.py
```
The script reports standard retrieval metrics in both directions (Hit@K, NDCG@K, MRR).

baseline:
```bash
python evaluate_random_baseline.py
```
# 6.Gradio demo

Set CHECKPOINT_PATH in app.py, then run:
```bash
python app.py
```
Default port: 7860.

# Contributors
Hongze Zhu, Haoyuan Yue,and Ruoyu Wang
zhuhongze@westlake.edu.cn | yuehaoyuan@westlake.edu.cn | wangruoyu@westlake.edu.cn 

