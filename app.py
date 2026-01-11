import os
import sys
import torch
import gradio as gr
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# RDKit 导入保护
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    HAS_RDKIT_DRAW = True
except ImportError:
    print("⚠️ Warning: RDKit drawing dependencies missing. Molecule visualization disabled.")
    HAS_RDKIT_DRAW = False

# 添加路径以便导入项目模块
sys.path.append('.')

# 导入你的模型和工具
from utils.module_loader import load_model
from dual_tower_recommendation_model import DualTowerRecommendationModel

# ==========================================
# 0. 自动生成演示数据 (关键新增)
# ==========================================
def create_demo_files():
    """在启动时自动创建演示用的 CSV 文件"""
    print("📝 Generating Demo CSV files...")
    
    # 1. 创建演示分子库 (demo_molecules.csv)
    # 包含：阿司匹林, 对乙酰氨基酚, 布洛芬, 咖啡因, 伊马替尼
    mol_data = {
        "name": ["Aspirin", "Paracetamol", "Ibuprofen", "Caffeine", "Imatinib"],
        "smiles": [
            "CC(=O)Oc1ccccc1C(=O)O", 
            "CC(=O)Nc1ccc(O)cc1", 
            "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
            "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"
        ]
    }
    pd.DataFrame(mol_data).to_csv("demo_molecules.csv", index=False)

    # 2. 创建演示蛋白库 (demo_proteins.csv)
    # 包含：简单的短序列用于测试
    prot_data = {
        "name": ["Example_Target_A", "Example_Target_B", "Example_Target_C"],
        "sequence": [
            "MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERGRRG", # 随机短序列
            "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET", # KRAS 片段
            "MALEKPSLAPGWEKGFCSSSPGNSPTPAPSSLTSSVREESPPGSPPPQPP"  # p53 片段
        ]
    }
    pd.DataFrame(prot_data).to_csv("demo_proteins.csv", index=False)
    print("✅ Demo files created: 'demo_molecules.csv' & 'demo_proteins.csv'")

# 执行生成
create_demo_files()

# ==========================================
# 1. 辅助类与函数
# ==========================================

class SimpleSMILESTokenizer:
    def __init__(self, vocab_size=128, max_length=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chars = sorted(list(" #%()+,-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnorsu"))
        self.char_to_idx = {c: i+1 for i, c in enumerate(self.chars)}
        
    def __call__(self, smiles):
        if not smiles or not isinstance(smiles, str):
            return torch.zeros(self.max_length, dtype=torch.long)
        token_ids = [self.char_to_idx.get(c, 1) for c in smiles[:self.max_length]]
        if len(token_ids) < self.max_length:
            token_ids += [0] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long)

def patch_sa_prot_sequence(seq):
    if not seq: return ""
    seq = "".join([c for c in seq if c.isalpha()])
    return " ".join([f"{aa}c" for aa in seq])

def get_molecule_image(smiles):
    if not HAS_RDKIT_DRAW: return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol: return Draw.MolToImage(mol, size=(300, 300))
    except: pass
    return None

# ==========================================
# 2. 模型加载
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG_PATH = "dual_tower_baseline.yaml"
CHECKPOINT_PATH = "/home/zhuhongze/Fajie/lightning_logs/version_19/checkpoints/best-checkpoint-epoch=15-valid_loss=3.67.ckpt"

print(f"🚀 Loading Model from {CHECKPOINT_PATH}...")
model = DualTowerRecommendationModel.load_from_checkpoint(CHECKPOINT_PATH, weights_only=False)
model.to(DEVICE)
model.eval()

print("📚 Loading Tokenizers...")
prot_tokenizer = AutoTokenizer.from_pretrained(model.hparams.protein_config_path, trust_remote_code=True)
mol_tokenizer = SimpleSMILESTokenizer(
    vocab_size=model.hparams.molecular_vocab_size,
    max_length=model.hparams.molecular_max_length
)

# ==========================================
# 3. 推理核心逻辑
# ==========================================

@torch.no_grad()
def calculate_score_p2m(protein_seq, smiles_list):
    patched_seq = patch_sa_prot_sequence(protein_seq)
    prot_inputs = prot_tokenizer(patched_seq, return_tensors="pt", padding="max_length", max_length=1024, truncation=True)
    prot_inputs = {k: v.to(DEVICE) for k, v in prot_inputs.items()}
    prot_emb = model(protein_inputs=prot_inputs)['protein']

    scores = []
    batch_size = 32
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i : i+batch_size]
        mol_ids = [mol_tokenizer(s) for s in batch_smiles]
        mol_ids = torch.stack(mol_ids).to(DEVICE)
        mol_inputs = {"molecule_input_ids": mol_ids, "molecule_attention_mask": (mol_ids != 0).long()}
        mol_emb = model(molecular_inputs=mol_inputs)['molecular']
        sim = torch.matmul(prot_emb, mol_emb.T) * logit_scale
        scores.extend(sim.squeeze(0).cpu().numpy().tolist())
    return scores

@torch.no_grad()
def calculate_score_m2p(smiles, protein_list):
    mol_ids = mol_tokenizer(smiles).unsqueeze(0).to(DEVICE)
    mol_inputs = {"molecule_input_ids": mol_ids, "molecule_attention_mask": (mol_ids != 0).long()}
    mol_emb = model(molecular_inputs=mol_inputs)['molecular']

    scores = []
    batch_size = 16
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    
    for i in range(0, len(protein_list), batch_size):
        batch_seqs = protein_list[i : i+batch_size]
        batch_patched = [patch_sa_prot_sequence(s) for s in batch_seqs]
        prot_inputs = prot_tokenizer(batch_patched, return_tensors="pt", padding=True, max_length=1024, truncation=True)
        prot_inputs = {k: v.to(DEVICE) for k, v in prot_inputs.items()}
        prot_emb = model(protein_inputs=prot_inputs)['protein']
        sim = torch.matmul(mol_emb, prot_emb.T) * logit_scale
        scores.extend(sim.squeeze(0).cpu().numpy().tolist())
    return scores

# ==========================================
# 4. Gradio 界面函数
# ==========================================

def predict_single_pair(protein_seq, smiles):
    if not protein_seq or not smiles: return "Input Missing", None
    try:
        scores = calculate_score_p2m(protein_seq, [smiles])
        img = get_molecule_image(smiles)
        return f"Binding Score: {scores[0]:.4f}", img
    except Exception as e:
        return f"Error: {str(e)}", None

def virtual_screening_p2m(protein_seq, csv_file):
    if not protein_seq or csv_file is None: return None, "Input Missing"
    try:
        df = pd.read_csv(csv_file.name)
        target_col = next((c for c in df.columns if c.lower() in ['smiles', 'smi']), None)
        if not target_col: return None, "CSV must contain 'smiles' column"
        
        smiles_list = df[target_col].astype(str).tolist()
        scores = calculate_score_p2m(protein_seq, smiles_list)
        df['score'] = scores
        df = df.sort_values(by='score', ascending=False)
        output_csv = "p2m_results.csv"
        df.to_csv(output_csv, index=False)
        return output_csv, f"✅ Processed {len(df)} molecules. Top 1: {df.iloc[0]['name'] if 'name' in df else ''} ({df.iloc[0]['score']:.4f})"
    except Exception as e:
        return None, f"Error: {str(e)}"

def target_fishing_m2p(smiles, csv_file):
    if not smiles or csv_file is None: return None, "Input Missing"
    try:
        df = pd.read_csv(csv_file.name)
        target_col = next((c for c in df.columns if c.lower() in ['sequence', 'seq', 'protein', 'aa_seq']), None)
        if not target_col: return None, "CSV must contain 'sequence' column"
        
        prot_list = df[target_col].astype(str).tolist()
        scores = calculate_score_m2p(smiles, prot_list)
        df['score'] = scores
        df = df.sort_values(by='score', ascending=False)
        output_csv = "m2p_results.csv"
        df.to_csv(output_csv, index=False)
        return output_csv, f"✅ Processed {len(df)} proteins. Top 1: {df.iloc[0]['name'] if 'name' in df else ''} ({df.iloc[0]['score']:.4f})"
    except Exception as e:
        return None, f"Error: {str(e)}"

# ==========================================
# 5. 构建 UI (含 Examples)
# ==========================================

with gr.Blocks(title="Fajie - AI Drug Discovery") as demo:
    gr.Markdown("# 🧬 Fajie: AI Drug Discovery System")
    gr.Markdown("Bi-directional Prediction with Dual-Tower SaProt + Transformer")
    
    with gr.Tabs():
        # --- Tab 1: Single Pair ---
        with gr.TabItem("1. Single Pair Scoring"):
            with gr.Row():
                with gr.Column():
                    t1_prot = gr.Textbox(lines=5, label="Protein Sequence")
                    t1_mol = gr.Textbox(lines=2, label="Molecule SMILES")
                    t1_btn = gr.Button("Predict", variant="primary")
                with gr.Column():
                    t1_res = gr.Textbox(label="Result")
                    t1_img = gr.Image(label="Molecule")
            t1_btn.click(predict_single_pair, inputs=[t1_prot, t1_mol], outputs=[t1_res, t1_img])
            
            # Tab 1 Examples
            gr.Examples(
                examples=[
                    ["MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERGRRG", "CC(=O)Oc1ccccc1C(=O)O"],
                    ["MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGET", "CN(C)C(=O)Cc1ccc(C)cc1"]
                ],
                inputs=[t1_prot, t1_mol],
                label="Click to use Example Data"
            )

        # --- Tab 2: Virtual Screening (P2M) ---
        with gr.TabItem("2. Virtual Screening (P2M)"):
            gr.Markdown("Input **1 Protein** + **CSV of Molecules**")
            with gr.Row():
                with gr.Column():
                    t2_prot = gr.Textbox(lines=5, label="Target Protein Sequence")
                    t2_file = gr.File(label="Upload Molecules CSV")
                    t2_btn = gr.Button("Start Screening", variant="primary")
                with gr.Column():
                    t2_status = gr.Textbox(label="Status")
                    t2_out = gr.File(label="Download Results")
            t2_btn.click(virtual_screening_p2m, inputs=[t2_prot, t2_file], outputs=[t2_out, t2_status])
            
            # Tab 2 Examples (指向自动生成的 demo_molecules.csv)
            gr.Examples(
                examples=[
                    ["MAAVILESIFLKRSQQKKKTSPLNFKKRLFLLTVHKLSYYEYDFERGRRG", "demo_molecules.csv"]
                ],
                inputs=[t2_prot, t2_file],
                label="Click to load Example Protein & CSV"
            )

        # --- Tab 3: Target Fishing (M2P) ---
        with gr.TabItem("3. Target Fishing (M2P)"):
            gr.Markdown("Input **1 Molecule** + **CSV of Proteins**")
            with gr.Row():
                with gr.Column():
                    t3_mol = gr.Textbox(lines=2, label="Query Molecule SMILES")
                    t3_file = gr.File(label="Upload Proteins CSV")
                    t3_btn = gr.Button("Start Fishing", variant="primary")
                with gr.Column():
                    t3_status = gr.Textbox(label="Status")
                    t3_out = gr.File(label="Download Results")
            t3_btn.click(target_fishing_m2p, inputs=[t3_mol, t3_file], outputs=[t3_out, t3_status])

            # Tab 3 Examples (指向自动生成的 demo_proteins.csv)
            gr.Examples(
                examples=[
                    ["CC(=O)Oc1ccccc1C(=O)O", "demo_proteins.csv"]
                ],
                inputs=[t3_mol, t3_file],
                label="Click to load Example Molecule & CSV"
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)