import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
# 假设 dataset 文件就在当前目录下，或者 python path 能找到
from protein_molecule_pairs_dataset import ProteinMoleculeDataset
import yaml
import os

# 🟢 配置文件路径
CONFIG_PATH = "dual_tower_baseline.yaml" 

def load_config_as_dict(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def decode_smiles(ids, idx_to_char):
    # 将 ID 解码回字符
    chars = []
    for i in ids:
        idx = i.item()
        if idx == 0: break # Padding
        if idx in idx_to_char:
            chars.append(idx_to_char[idx])
        else:
            chars.append("[UNK]")
    return "".join(chars)

def main():
    print(f">>> Loading Configuration from {CONFIG_PATH}...")
    try:
        raw_config = load_config_as_dict(CONFIG_PATH)
    except FileNotFoundError:
        print(f"❌ Error: File not found at {CONFIG_PATH}")
        return

    # 🟢 1. 精准提取路径 (针对你的 YAML 结构)
    dataset_cfg = raw_config.get('dataset', {})
    
    # 获取 LMDB 路径
    train_lmdb = dataset_cfg.get('train_lmdb')
    
    if not train_lmdb:
        print("❌ Error: Could not find 'dataset -> train_lmdb' in yaml.")
        return
    
    print(f">>> Found Dataset Path: {train_lmdb}")

    # 获取 Kwargs 参数
    dataset_kwargs = dataset_cfg.get('kwargs', {})
    
    protein_tokenizer_path = dataset_kwargs.get('protein_tokenizer')
    max_protein_len = dataset_kwargs.get('max_protein_length', 1024)
    max_molecule_len = dataset_kwargs.get('max_molecular_length', 512)
    mol_vocab_size = dataset_kwargs.get('molecular_vocab_size', 128)
    
    print(f">>> Tokenizer: {protein_tokenizer_path}")
    print(f">>> Max Protein Len: {max_protein_len}")
    print(f">>> Max Molecule Len: {max_molecule_len}")
    
    if not protein_tokenizer_path:
        print("❌ Error: Could not find 'dataset -> kwargs -> protein_tokenizer' in yaml.")
        return

    # 🟢 2. 实例化 Dataset
    print(f"\n>>> Instantiating Dataset...")
    try:
        dataset = ProteinMoleculeDataset(
            lmdb_path=train_lmdb,
            protein_tokenizer_path=protein_tokenizer_path,
            max_protein_len=max_protein_len,
            max_molecule_len=max_molecule_len,
            molecule_vocab_size=mol_vocab_size
        )
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        # 尝试打印更详细的错误
        import traceback
        traceback.print_exc()
        return
    
    # 3. 准备解码字典
    prot_tokenizer = dataset.protein_tokenizer
    idx_to_char = {v: k for k, v in dataset.char_to_idx.items()}
    
    print("\n>>> Checking first 5 samples...")
    print("="*80)
    
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    for i, batch in enumerate(loader):
        if i >= 5: break
        
        prot_ids = batch["protein_input_ids"][0]
        mol_ids = batch["molecule_input_ids"][0]
        
        # --- 检查蛋白质 ---
        # 统计有效长度（不包含 Pad）
        # Pad token id 通常是 1，但也可能是 0，根据 tokenizer 而定
        pad_id = prot_tokenizer.pad_token_id
        prot_len = (prot_ids != pad_id).sum().item()
        
        # 解码回字符串
        prot_str = prot_tokenizer.decode(prot_ids, skip_special_tokens=True)
        # 检查是否有大量 UNK (UNK token id 通常是 3)
        unk_id = prot_tokenizer.unk_token_id
        unk_count = (prot_ids == unk_id).sum().item()
        
        # --- 检查分子 ---
        mol_len = (mol_ids != 0).sum().item()
        mol_str = decode_smiles(mol_ids, idx_to_char)
        mol_unk_count = mol_str.count("[UNK]")
        
        print(f"Sample {i+1}:")
        print(f"[Protein]")
        print(f"  - Valid Length: {prot_len} / {max_protein_len}")
        print(f"  - UNK Count: {unk_count} (Should be 0)")
        print(f"  - Raw IDs[:10]: {prot_ids[:10].tolist()}")
        print(f"  - Preview: {prot_str[:100]}...")
        
        print(f"[Molecule]")
        print(f"  - Valid Length: {mol_len} / {max_molecule_len}")
        print(f"  - UNK Count: {mol_unk_count} (Should be 0)")
        print(f"  - Raw String: {mol_str}")
        
        print("-" * 80)
        
        # 🚨 诊断逻辑
        if unk_count > prot_len * 0.5:
            print("⚠️ 警告：蛋白质序列大部分是 UNK！这说明 Tokenizer 和数据不匹配！")
            print("   (ESM Tokenizer 通常不识别小写字母或特殊字符)")
        
        if prot_len < 5:
            print("⚠️ 警告：蛋白质序列极短或为空！")
            
        if mol_len < 2:
            print("⚠️ 警告：分子序列极短或为空！")

if __name__ == "__main__":
    main()