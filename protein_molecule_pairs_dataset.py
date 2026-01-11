import torch
import lmdb
import json
import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class ProteinMoleculeDataset(Dataset):
    def __init__(self, lmdb_path, protein_tokenizer_path=None, max_protein_len=1024, max_molecule_len=512, molecule_vocab_size=128):
        super().__init__()
        self.lmdb_path = lmdb_path
        self.env = None
        self.max_protein_len = max_protein_len
        self.max_molecule_len = max_molecule_len
        
        # SMILES 字典
        self.smiles_chars = [
            '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'V',
            'Z', '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't', 'u'
        ]
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.smiles_chars)}
        self.unk_idx = len(self.smiles_chars) + 1

        # 加载 Tokenizer
        if protein_tokenizer_path:
            # print(f"[Dataset] Loading Tokenizer from: {protein_tokenizer_path}")
            self.protein_tokenizer = AutoTokenizer.from_pretrained(protein_tokenizer_path, trust_remote_code=True)
        else:
            raise ValueError("protein_tokenizer_path must be provided!")

        # 检查 LMDB
        if not os.path.exists(lmdb_path):
             raise FileNotFoundError(f"LMDB path not found: {lmdb_path}")

        # 读取长度
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = int(txn.get(b"length").decode('utf-8'))
        env.close()

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        with self.env.begin(write=False) as txn:
            json_str = txn.get(str(index).encode('utf-8')).decode('utf-8')
            item = json.loads(json_str)
            
        # 🟢 这里的 protein_seq 已经是我们在 LMDB 里存好的完美格式 "M#d V#d ..."
        # 不要再对它做任何 replace 或 join 操作！
        protein_seq = item['protein_seq']
        molecule_smiles = item['molecule_smiles']

        # 1. 处理蛋白质 (SaProt Tokenizer 直接吃)
        prot_inputs = self.protein_tokenizer(
            protein_seq, 
            max_length=self.max_protein_len, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        prot_ids = prot_inputs['input_ids'].squeeze(0)
        prot_mask = prot_inputs['attention_mask'].squeeze(0)

        # 2. 处理分子
        mol_ids = torch.zeros(self.max_molecule_len, dtype=torch.long)
        smiles_str = molecule_smiles[:self.max_molecule_len]
        
        for i, char in enumerate(smiles_str):
            if char in self.char_to_idx:
                mol_ids[i] = self.char_to_idx[char]
            else:
                mol_ids[i] = self.unk_idx
            
        return {
            "protein_input_ids": prot_ids,
            "protein_attention_mask": prot_mask,
            "molecule_input_ids": mol_ids,
            "molecule_attention_mask": (mol_ids > 0).long()
        }

class ProteinMoleculePairDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.dataloader_kwargs.batch_size
        self.num_workers = config.dataloader_kwargs.num_workers
        
    def setup(self, stage=None):
        kwargs = self.config.kwargs
        mol_vocab_size = kwargs.get('molecular_vocab_size', 128)
        
        self.train_dataset = ProteinMoleculeDataset(
            self.config.train_lmdb, kwargs.protein_tokenizer, kwargs.max_protein_length, kwargs.max_molecular_length, mol_vocab_size
        )
        self.valid_dataset = ProteinMoleculeDataset(
            self.config.valid_lmdb, kwargs.protein_tokenizer, kwargs.max_protein_length, kwargs.max_molecular_length, mol_vocab_size
        )
        self.test_dataset = ProteinMoleculeDataset(
            self.config.test_lmdb, kwargs.protein_tokenizer, kwargs.max_protein_length, kwargs.max_molecular_length, mol_vocab_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)