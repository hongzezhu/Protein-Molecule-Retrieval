import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from transformers import AutoModel

class DualTowerRecommendationModel(pl.LightningModule):
    def __init__(self,
                 protein_config_path: str,
                 protein_load_pretrained: bool = True,
                 foldseek_path: str = None,
                 plddt_threshold: float = 70.0,
                 protein_unfreeze_last_n_layers: int = 2, # 默认解冻最后2层
                 
                 molecular_vocab_size: int = 128,
                 molecular_embed_dim: int = 512, 
                 molecular_num_layers: int = 4,  
                 molecular_num_heads: int = 8,
                 molecular_ffn_dim: int = None,
                 molecular_max_length: int = 512,
                 molecular_dropout: float = 0.1, 
                 
                 temperature: float = 0.07, 
                 embedding_dim: int = 1024, 
                 
                 lr_scheduler_kwargs: dict = None,
                 optimizer_kwargs: dict = None,
                 save_path: str = None,
                 from_checkpoint: str = None,
                 load_prev_scheduler: bool = False,
                 save_weights_only: bool = True,
                 **kwargs): 
        
        super().__init__()
        self.save_hyperparameters() 
        
        self.protein_config_path = protein_config_path
        self.protein_unfreeze_last_n_layers = protein_unfreeze_last_n_layers
        
        # Configs
        self.molecular_vocab_size = molecular_vocab_size
        self.molecular_embed_dim = molecular_embed_dim
        self.molecular_num_layers = molecular_num_layers
        self.molecular_num_heads = molecular_num_heads
        self.molecular_max_length = molecular_max_length
        self.embedding_dim = embedding_dim
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        # Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
        self.initialize_model()
    
    def initialize_model(self):
        print(f"[Model] Loading Protein Model from: {self.protein_config_path}")
        self.protein_encoder = AutoModel.from_pretrained(self.protein_config_path, trust_remote_code=True)
        protein_embed_dim = self.protein_encoder.config.hidden_size
        
        # 🟢 [解冻策略]
        # 1. 先冻结所有层
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
            
        # 2. 如果指定了解冻层数，解冻最后 N 层
        if self.protein_unfreeze_last_n_layers > 0:
            print(f"🔓 Unfreezing last {self.protein_unfreeze_last_n_layers} layers of Protein Encoder...")
            if hasattr(self.protein_encoder, 'encoder') and hasattr(self.protein_encoder.encoder, 'layer'):
                for layer in self.protein_encoder.encoder.layer[-self.protein_unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
            if hasattr(self.protein_encoder, 'pooler') and self.protein_encoder.pooler is not None:
                for param in self.protein_encoder.pooler.parameters():
                    param.requires_grad = True
        
        # 2. Molecular Tower (Transformer)
        self.molecular_token_embedding = nn.Embedding(self.molecular_vocab_size, self.molecular_embed_dim)
        self.molecular_position_embedding = nn.Embedding(self.molecular_max_length, self.molecular_embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.molecular_embed_dim,
            nhead=self.molecular_num_heads,
            dim_feedforward=4 * self.molecular_embed_dim,
            dropout=0.1,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )
        self.molecular_transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.molecular_num_layers)
        
        # 🟢 Projection Head (SimCLR v2 MLP+BN)
        self.protein_proj = nn.Sequential(
            nn.Linear(protein_embed_dim, protein_embed_dim, bias=False),
            nn.BatchNorm1d(protein_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(protein_embed_dim, protein_embed_dim, bias=False),
            nn.BatchNorm1d(protein_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(protein_embed_dim, self.embedding_dim, bias=False),
            nn.BatchNorm1d(self.embedding_dim)
        )

        self.molecular_proj = nn.Sequential(
            nn.Linear(self.molecular_embed_dim, self.molecular_embed_dim, bias=False),
            nn.BatchNorm1d(self.molecular_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.molecular_embed_dim, self.molecular_embed_dim, bias=False),
            nn.BatchNorm1d(self.molecular_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.molecular_embed_dim, self.embedding_dim, bias=False),
            nn.BatchNorm1d(self.embedding_dim)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight) 
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # 🟢 补回缺失的辅助函数 (用于分阶段训练)
    def freeze_protein_tower(self):
        # 恢复到初始化时的状态 (保持部分解冻或全冻结)
        for param in self.protein_encoder.parameters():
            param.requires_grad = False
        
        # 只有在非 warmup 阶段，才恢复部分解冻
        if self.protein_unfreeze_last_n_layers > 0:
             if hasattr(self.protein_encoder, 'encoder') and hasattr(self.protein_encoder.encoder, 'layer'):
                for layer in self.protein_encoder.encoder.layer[-self.protein_unfreeze_last_n_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
    
    def unfreeze_molecular_tower(self):
        for m in [self.molecular_token_embedding, self.molecular_position_embedding, self.molecular_transformer]:
            for param in m.parameters(): param.requires_grad = True

    def freeze_molecular_tower(self):
        for m in [self.molecular_token_embedding, self.molecular_position_embedding, self.molecular_transformer]:
            for param in m.parameters(): param.requires_grad = False
            
    def unfreeze_projection_layers(self):
        for m in [self.protein_proj, self.molecular_proj]:
            for param in m.parameters(): param.requires_grad = True

    def set_training_phase(self, phase: str):
        print(f"\n[Training Phase] Setting to: {phase}")
        if phase == "projection_warmup":
            # 预热阶段：冻结所有 Encoder，只训练 Projection Head
            for param in self.protein_encoder.parameters(): param.requires_grad = False
            self.freeze_molecular_tower()
            self.unfreeze_projection_layers()
        elif phase == "molecule_training":
            # 正式训练：恢复配置的解冻状态
            self.freeze_protein_tower() # 这会根据 unfreeze_last_n_layers 决定是否解冻 ESM
            self.unfreeze_molecular_tower()
            self.unfreeze_projection_layers()
        else:
            print(f"Unknown phase {phase}, default to standard training.")

    def forward(self, protein_inputs=None, molecular_inputs=None):
        embeddings = {}
        
        if protein_inputs is not None:
            outputs = self.protein_encoder(**protein_inputs)
            hidden_state = outputs.last_hidden_state
            
            attention_mask = protein_inputs['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            protein_repr = sum_embeddings / sum_mask
            
            protein_emb = self.protein_proj(protein_repr)
            protein_emb = F.normalize(protein_emb, p=2, dim=1)
            embeddings["protein"] = protein_emb
            embeddings["protein_ids"] = protein_inputs["input_ids"]
        
        if molecular_inputs is not None:
            input_ids = molecular_inputs["molecule_input_ids"]
            attention_mask = molecular_inputs.get("molecule_attention_mask")
            input_ids = torch.clamp(input_ids, max=self.molecular_vocab_size - 1)
            
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            x = self.molecular_token_embedding(input_ids) + self.molecular_position_embedding(position_ids)
            
            attn_mask = (1 - attention_mask).bool() if attention_mask is not None else None
            encoded = self.molecular_transformer(x, src_key_padding_mask=attn_mask)
            
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                encoded = encoded * mask_expanded
                pooled = encoded.sum(dim=1) / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            else:
                pooled = encoded.mean(dim=1)
            
            molecular_emb = self.molecular_proj(pooled)
            molecular_emb = F.normalize(molecular_emb, p=2, dim=1)
            embeddings["molecular"] = molecular_emb
        
        return embeddings

    def loss_func(self, stage: str, outputs, labels=None):
        protein_emb = outputs["protein"]
        molecular_emb = outputs["molecular"]
        
        # 1. 计算相似度矩阵 (Batch x Batch)
        # 限制 logit_scale 最大值防止梯度爆炸，最小值防止分布过于平滑
        logit_scale = self.logit_scale.exp().clamp(min=1e-2, max=100.0)
        sim_p2m = torch.matmul(protein_emb, molecular_emb.T) * logit_scale
        sim_m2p = sim_p2m.T

        # 2. 构建目标矩阵 (Targets)
        # 只要 Protein ID 相同，就认为是正样本 (1.0)，否则是负样本 (0.0)
        prot_ids = outputs["protein_ids"]
        # shape: (Batch, Batch), bool
        is_same_protein = (prot_ids.unsqueeze(1) == prot_ids.unsqueeze(0)).all(dim=-1)        
        # 转换为 float 类型作为 Soft Labels
        targets = is_same_protein.float().to(protein_emb.device)

        # 3. 归一化 Targets (Multi-Positive Softmax Cross Entropy)
        # 如果一行有 3 个正样本，每个正样本的概率目标变成 1/3
        # 这样 loss = - sum( target_prob * log(pred_prob) )
        p2m_targets = targets / targets.sum(dim=1, keepdim=True)
        m2p_targets = targets.T / targets.T.sum(dim=1, keepdim=True)

        # 4. 计算 Loss (手动实现 Cross Entropy 以支持 Soft Labels)
        # F.log_softmax 计算预测概率的 log 值
        log_probs_p2m = F.log_softmax(sim_p2m, dim=1)
        log_probs_m2p = F.log_softmax(sim_m2p, dim=1)

        # loss = - sum(target * log_prob)
        loss_p2m = -torch.sum(p2m_targets * log_probs_p2m, dim=1).mean()
        loss_m2p = -torch.sum(m2p_targets * log_probs_m2p, dim=1).mean()
        
        loss = (loss_p2m + loss_m2p) / 2
        
        # 5. Logging & Debug
        if self.global_step % 10 == 0:
            # 统计一下平均每个样本有多少个正样本（除了它自己）
            avg_positives = (targets.sum(dim=1) - 1).mean().item()
            with torch.no_grad():
                # 简单的准确率估算：看最大的 logit 是否落在正样本集合内
                # P2M Accuracy
                max_idx_p2m = sim_p2m.argmax(dim=1) # (Batch,)
                # gather 获取 max_idx 对应的 target 值 (是1就是对，0就是错)
                acc_p2m = torch.gather(targets, 1, max_idx_p2m.unsqueeze(1)).mean().item()
                
            print(f"\n[Step {self.global_step}] Loss: {loss:.4f} | Avg Dupes per batch: {avg_positives:.1f} | Batch Acc: {acc_p2m:.2%}")

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        inputs = {
            "protein_inputs": {"input_ids": batch["protein_input_ids"], "attention_mask": batch["protein_attention_mask"]},
            "molecular_inputs": {"molecule_input_ids": batch["molecule_input_ids"], "molecule_attention_mask": batch["molecule_attention_mask"]}
        }
        outputs = self.forward(**inputs)
        return self.loss_func('train', outputs)

    def validation_step(self, batch, batch_idx):
        inputs = {
            "protein_inputs": {"input_ids": batch["protein_input_ids"], "attention_mask": batch["protein_attention_mask"]},
            "molecular_inputs": {"molecule_input_ids": batch["molecule_input_ids"], "molecule_attention_mask": batch["molecule_attention_mask"]}
        }
        outputs = self.forward(**inputs)
        return self.loss_func('valid', outputs)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        lr = self.lr_scheduler_kwargs.get('max_lr', 1e-4) 
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=self.lr_scheduler_kwargs.get('weight_decay', 0.01))
        
        warmup_steps = self.lr_scheduler_kwargs.get('warmup_steps', 100)
        total_steps = 2000 
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return {
            "optimizer": optimizer, 
            "lr_scheduler": {"scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda), "interval": "step"}
        }