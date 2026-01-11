import os
import sys
import yaml
import torch
import numpy as np
import math
from tqdm import tqdm
from easydict import EasyDict
from collections import defaultdict

# 添加当前目录到系统路径
sys.path.append('.')

# 🟢 导入 load_model 用于初始化未经训练的模型
from utils.module_loader import load_dataset, load_model
from dual_tower_recommendation_model import DualTowerRecommendationModel

def calculate_ndcg(hits, k, num_positives):
    """
    计算 NDCG@K
    """
    dcg = 0.0
    current_hits = hits[:k]
    for i, rel in enumerate(current_hits):
        if rel > 0:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    num_possible_hits = min(k, num_positives)
    for i in range(num_possible_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    if idcg == 0: return 0.0
    return dcg / idcg

def run_retrieval_task(query_group, gallery_embs, task_name, device, logit_scale):
    """
    通用的检索任务运行函数 (与 evaluate_grouped.py 保持一致)
    """
    print(f"\n🚀 Running {task_name} Task (Random Baseline)...")
    print(f"   Queries: {len(query_group)} | Gallery Size: {gallery_embs.shape[0]}")
    
    metrics_sum = {
        "Hit@1": 0, "Hit@5": 0, "Hit@10": 0, "Hit@50": 0,
        "P@1": 0, "P@5": 0, "P@10": 0, "P@50": 0,
        "NDCG@10": 0, "NDCG@50": 0,
        "MRR": 0
    }
    
    n_queries = len(query_group)
    TOP_K_SEARCH = 100 
    
    # 转到 GPU
    gallery_embs = gallery_embs.to(device)
    
    for q_hash, data in tqdm(query_group.items(), desc=f"{task_name}"):
        q_emb = data['emb'].to(device).unsqueeze(0)
        pos_indices = set(data['pos_indices'])
        num_pos = len(pos_indices)
        
        # 相似度计算
        sims = torch.matmul(q_emb, gallery_embs.T).squeeze(0) * logit_scale
        
        # Top-K
        _, topk_indices = torch.topk(sims, k=TOP_K_SEARCH)
        topk_indices = topk_indices.cpu().tolist()
        
        hits = [1 if idx in pos_indices else 0 for idx in topk_indices]
        
        # Metrics Calculation
        if sum(hits[:1]) > 0: metrics_sum["Hit@1"] += 1
        if sum(hits[:5]) > 0: metrics_sum["Hit@5"] += 1
        if sum(hits[:10]) > 0: metrics_sum["Hit@10"] += 1
        if sum(hits[:50]) > 0: metrics_sum["Hit@50"] += 1

        metrics_sum["P@1"] += sum(hits[:1]) / 1.0
        metrics_sum["P@5"] += sum(hits[:5]) / 5.0
        metrics_sum["P@10"] += sum(hits[:10]) / 10.0
        metrics_sum["P@50"] += sum(hits[:50]) / 50.0

        metrics_sum["NDCG@10"] += calculate_ndcg(hits, 10, num_pos)
        metrics_sum["NDCG@50"] += calculate_ndcg(hits, 50, num_pos)

        try:
            first_hit_rank = hits.index(1) + 1
            metrics_sum["MRR"] += 1.0 / first_hit_rank
        except ValueError:
            metrics_sum["MRR"] += 0.0

    # Print Results
    print("\n" + "="*50)
    print(f"🎲 {task_name} Results (Queries: {n_queries})")
    print("="*50)
    
    def print_metric(name, key, is_percent=True):
        val = metrics_sum[key] / n_queries
        fmt = f"{val:.2%}" if is_percent else f"{val:.4f}"
        print(f"{name:<20} : {fmt}")

    print_metric("Hit Rate @ 1", "Hit@1")
    print_metric("Hit Rate @ 10", "Hit@10")
    print_metric("Precision @ 10", "P@10")
    print_metric("NDCG @ 10", "NDCG@10", is_percent=False)
    print_metric("MRR", "MRR", is_percent=False)
    print("="*50)

def evaluate_untrained_bidirectional(config_path):
    # 1. 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = EasyDict(yaml.safe_load(f))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # 🟢 初始化未经训练的模型 (Random Baseline)
    print("🎲 Initializing UNTRAINED model (Random Weights)...")
    model = load_model(config.model)
    model.to(device)
    model.eval()

    # 2. 加载数据
    print("📚 Loading Test Data...")
    config.dataset.test_lmdb = "LMDB/BindingDB_Filtered/test"
    data_module = load_dataset(config.dataset)
    data_module.setup(stage='test')
    test_loader = data_module.test_dataloader()

    # 3. 提取特征 (双向)
    print("\n⚡ Extracting features for BOTH directions (Random Baseline)...")
    
    unique_prot_embs = [] 
    unique_mol_embs = []  
    
    prot_hash_to_idx = {} 
    mol_hash_to_idx = {}
    
    p2m_group = defaultdict(lambda: {'emb': None, 'pos_indices': []})
    m2p_group = defaultdict(lambda: {'emb': None, 'pos_indices': []})
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extraction"):
            # 🟢 [关键修复] 手动指定 Key
            prot_inputs = {
                "input_ids": batch["protein_input_ids"].to(device),
                "attention_mask": batch["protein_attention_mask"].to(device)
            }
            mol_inputs = {
                "molecule_input_ids": batch["molecule_input_ids"].to(device),
                "molecule_attention_mask": batch["molecule_attention_mask"].to(device)
            }
            
            p_out = model(protein_inputs=prot_inputs)['protein'].cpu()
            m_out = model(molecular_inputs=mol_inputs)['molecular'].cpu()
            
            p_input_ids = batch['protein_input_ids'].cpu().numpy()
            m_input_ids = batch['molecule_input_ids'].cpu().numpy()
            
            batch_size = p_out.shape[0]
            
            for i in range(batch_size):
                # --- 处理 Protein ---
                p_hash = bytes(p_input_ids[i]).hex()
                if p_hash not in prot_hash_to_idx:
                    prot_hash_to_idx[p_hash] = len(unique_prot_embs)
                    unique_prot_embs.append(p_out[i])
                p_idx = prot_hash_to_idx[p_hash]
                
                # --- 处理 Molecule ---
                m_hash = bytes(m_input_ids[i]).hex()
                if m_hash not in mol_hash_to_idx:
                    mol_hash_to_idx[m_hash] = len(unique_mol_embs)
                    unique_mol_embs.append(m_out[i])
                m_idx = mol_hash_to_idx[m_hash]
                
                # --- 构建 P2M ---
                if p2m_group[p_hash]['emb'] is None: p2m_group[p_hash]['emb'] = p_out[i]
                p2m_group[p_hash]['pos_indices'].append(m_idx)
                
                # --- 构建 M2P ---
                if m2p_group[m_hash]['emb'] is None: m2p_group[m_hash]['emb'] = m_out[i]
                m2p_group[m_hash]['pos_indices'].append(p_idx)

    gallery_mol_embs = torch.stack(unique_mol_embs)
    gallery_prot_embs = torch.stack(unique_prot_embs)
    
    logit_scale = 1.0 # 随机模型不需要 temperature
    
    print(f"✅ Unique Proteins: {len(unique_prot_embs)}")
    print(f"✅ Unique Molecules: {len(unique_mol_embs)}")
    
    # 4. 运行双向评估
    # Task 1: Virtual Screening (P2M)
    run_retrieval_task(p2m_group, gallery_mol_embs, "Virtual Screening (P2M)", device, logit_scale)

    # Task 2: Target Fishing (M2P)
    run_retrieval_task(m2p_group, gallery_prot_embs, "Target Fishing (M2P)", device, logit_scale)

if __name__ == "__main__":
    CONFIG = "dual_tower_baseline.yaml"
    # 随机基准不需要加载 checkpoint
    evaluate_untrained_bidirectional(CONFIG)