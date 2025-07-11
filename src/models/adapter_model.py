import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .pooling import Attention1dPoolingHead, MeanPoolingHead, LightAttentionPoolingHead
from .pooling import MeanPooling, MeanPoolingProjection
from .pooling import Attention1dPoolingRegressionHead, MeanPoolingRegressionHead, LightAttentionPoolingRegressionHead
from .pooling import MultiTaskRegressionProjection

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    cos = cos[:, :, : x.shape[-2], :]
    sin = sin[:, :, : x.shape[-2], :]

    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings based on those in
    [RoFormer](https://huggingface.co/docs/transformers/model_doc/roformer). Query and keys are transformed by rotation
    matrices which depend on their relative positions.
    """

    def __init__(self, dim: int):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        inv_freq = inv_freq
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=2):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k, seq_dimension=-2)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


class CrossModalAttention(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.attention_head_size = args.hidden_size // args.num_attention_head
        assert (
            self.attention_head_size * args.num_attention_head == args.hidden_size
        ), "Embed size needs to be divisible by num heads"
        self.num_attention_head = args.num_attention_head
        self.hidden_size = args.hidden_size
        
        self.query_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.key_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.value_proj = nn.Linear(args.hidden_size, args.hidden_size)
        
        self.dropout = nn.Dropout(args.attention_probs_dropout)
        
        self.out_proj = nn.Linear(args.hidden_size, args.hidden_size)
        self.rotary_embeddings = RotaryEmbedding(dim=self.attention_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_head, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, query, key, value, attention_mask=None, output_attentions=False):
        key_layer = self.transpose_for_scores(self.key_proj(key))
        value_layer = self.transpose_for_scores(self.value_proj(value))
        query_layer = self.transpose_for_scores(self.query_proj(query))
        query_layer = query_layer * self.attention_head_size**-0.5
        
        query_layer, key_layer = self.rotary_embeddings(query_layer, key_layer)
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else context_layer
        
        return outputs


class SeparatedRegressionHeads(nn.Module):
    """
    Separated regression heads, each task is modeled independently
    """

    def __init__(self, hidden_size, num_tasks=3, pooling_method='mean', 
                 pooling_dropout=0.1, shared_feature_layers=True):
        super().__init__()
        self.num_tasks = num_tasks
        self.pooling_method = pooling_method
        self.shared_feature_layers = shared_feature_layers
        
        # Shared feature extraction layers 
        if shared_feature_layers:
            self.shared_feature_extractor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(pooling_dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(pooling_dropout)
            )
            feature_dim = hidden_size // 2
        else:
            feature_dim = hidden_size
        
        # Create independent regression heads for each task
        self.task_heads = nn.ModuleList()
        task_names = ['phopt', 'phmin', 'phmax']
        
        for i, task_name in enumerate(task_names):
            # Specialized regression layers for each task
            if pooling_method == 'attention1d':
                # Use attention pooling
                pooling_layer = Attention1dPooling(feature_dim, pooling_dropout)
                regression_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.LayerNorm(feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 2, feature_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 4, 1)
                )
                task_head = nn.ModuleDict({
                    'pooling': pooling_layer,
                    'regression': regression_head
                })
            elif pooling_method == 'light_attention':
                # Use lightweight attention pooling
                pooling_layer = LightAttentionPooling(feature_dim, pooling_dropout)
                regression_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.LayerNorm(feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 2, feature_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 4, 1)
                )
                task_head = nn.ModuleDict({
                    'pooling': pooling_layer,
                    'regression': regression_head
                })
            else:  # mean pooling
                # Use mean pooling + complete regression network
                regression_head = nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.LayerNorm(feature_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 2, feature_dim // 4),
                    nn.LayerNorm(feature_dim // 4),
                    nn.ReLU(),
                    nn.Dropout(pooling_dropout),
                    nn.Linear(feature_dim // 4, 1)
                )
                task_head = regression_head
            
            self.task_heads.append(task_head)
    
    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_tasks]
        """

        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Optional shared feature extraction
        if self.shared_feature_layers:
            # Apply shared feature extraction to each time step
            features = hidden_states.view(-1, hidden_size)
            features = self.shared_feature_extractor(features)
            features = features.view(batch_size, seq_len, -1)
        else:
            features = hidden_states
        
        # Independent computation for each task
        task_outputs = []
        
        for i, task_head in enumerate(self.task_heads):
            if self.pooling_method in ['attention1d', 'light_attention']:
                # Has specialized pooling layers
                pooled = task_head['pooling'](features, attention_mask)
                task_output = task_head['regression'](pooled)
            else:
                # mean pooling + regression
                # Mean pooling
                pooled = (features * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
                task_output = task_head(pooled)
            
            task_outputs.append(task_output)
        
        # Concatenate outputs from all tasks
        logits = torch.cat(task_outputs, dim=1)  # [batch_size, num_tasks]
        
        return logits


class Attention1dPooling(nn.Module):
    """
    Simplified 1D attention pooling
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
            nn.Dropout(dropout)
        )
    
    def forward(self, hidden_states, attention_mask):
        # Calculate attention weights
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
        
        # Apply mask
        attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Weighted average
        pooled = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return pooled


class LightAttentionPooling(nn.Module):
    """
    Lightweight attention pooling
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.attention_linear = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, hidden_states, attention_mask):
        # Simple linear transformation + softmax
        attention_weights = self.attention_linear(hidden_states).squeeze(-1)
        attention_weights = attention_weights.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted average
        pooled = torch.sum(hidden_states * attention_weights.unsqueeze(-1), dim=1)
        return pooled


class AdapterModel(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
        if 'foldseek_seq' in args.structure_seq:
            self.foldseek_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_foldseek = CrossModalAttention(args)
        if 'ss8_seq' in args.structure_seq:
            self.ss_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_ss = CrossModalAttention(args)
        if 'esm3_structure_seq' in args.structure_seq:
            self.esm3_structure_embedding = nn.Embedding(args.vocab_size, args.hidden_size)
            self.cross_attention_esm3_structure = CrossModalAttention(args)
        
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        
        # Uniformly use problem_type to determine if it's multi-task regression
        is_multitask_regression = (args.problem_type == 'regression' and args.num_labels > 1)
        shared_layers = getattr(args, 'shared_task_layers', True)
        
        if is_multitask_regression:
            # Multi-task regression - prioritize using separated regression heads
            regression_head_type = getattr(args, 'regression_head_type', 'separated')
            
            if regression_head_type == 'separated':
                # Use separated regression heads (recommended)
                self.classifier = SeparatedRegressionHeads(
                    hidden_size=args.hidden_size,
                    num_tasks=args.num_labels,
                    pooling_method=args.pooling_method,
                    pooling_dropout=args.pooling_dropout,
                    shared_feature_layers=getattr(args, 'shared_feature_layers', True)
                )
            else:
                # Original unified regression head logic (as fallback)
                if args.pooling_method == 'attention1d':
                    self.classifier = Attention1dPoolingRegressionHead(
                        args.hidden_size, args.num_labels, args.pooling_dropout, shared_layers
                    )
                elif args.pooling_method == 'mean':
                    if "PPI" in args.dataset:
                        self.pooling = MeanPooling()
                        self.projection = MultiTaskRegressionProjection(
                            args.hidden_size, args.num_labels, args.pooling_dropout, shared_layers
                        )
                    else:
                        self.classifier = MeanPoolingRegressionHead(
                            args.hidden_size, args.num_labels, args.pooling_dropout, shared_layers
                        )
                elif args.pooling_method == 'light_attention':
                    self.classifier = LightAttentionPoolingRegressionHead(
                        args.hidden_size, args.num_labels, args.pooling_dropout, shared_layers=shared_layers
                    )
                else:
                    raise ValueError(f"Pooling method {args.pooling_method} not supported for regression")
        else:
            # classification logic
            if args.pooling_method == 'attention1d':
                self.classifier = Attention1dPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            elif args.pooling_method == 'mean':
                if "PPI" in args.dataset:
                    self.pooling = MeanPooling()
                    self.projection = MeanPoolingProjection(args.hidden_size, args.num_labels, args.pooling_dropout)
                else:
                    self.classifier = MeanPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            elif args.pooling_method == 'light_attention':
                self.classifier = LightAttentionPoolingHead(args.hidden_size, args.num_labels, args.pooling_dropout)
            else:
                raise ValueError(f"classifier method {args.pooling_method} not supported")
        
        # Improved weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Improved weight initialization
        """

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization, more friendly for regression tasks
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def plm_embedding(self, plm_model, aa_seq, attention_mask, structure_tokens=None):

        with torch.no_grad():
            if "ProSST" in self.args.plm_model:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, ss_input_ids=structure_tokens, output_hidden_states=True)
            elif "Prime" in self.args.plm_model or "deep" in self.args.plm_model:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask, output_hidden_states=True)
            elif self.training and hasattr(self, 'args') and self.args.training_method == 'full':
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
            else:
                outputs = plm_model(input_ids=aa_seq, attention_mask=attention_mask)
            if "ProSST" in self.args.plm_model or "Prime" in self.args.plm_model:
                seq_embeds = outputs.hidden_states[-1]
            else:
                seq_embeds = outputs.last_hidden_state
        gc.collect()
        torch.cuda.empty_cache()
        return seq_embeds
    
    def forward(self, plm_model, batch):

        if "ProSST" in self.args.plm_model:
            aa_seq, attention_mask, stru_tokens = batch['aa_seq_input_ids'], batch['aa_seq_attention_mask'], batch['aa_seq_stru_tokens']
            seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask, stru_tokens)
        else:
            aa_seq, attention_mask = batch['aa_seq_input_ids'], batch['aa_seq_attention_mask']
            seq_embeds = self.plm_embedding(plm_model, aa_seq, attention_mask)

        # embeds
        embeds = seq_embeds

        if 'foldseek_seq' in self.args.structure_seq:
            foldseek_seq = batch['foldseek_seq_input_ids']
            foldseek_embeds = self.foldseek_embedding(foldseek_seq)
            foldseek_embeds = self.cross_attention_foldseek(foldseek_embeds, seq_embeds, seq_embeds, attention_mask)
            embeds = seq_embeds + foldseek_embeds
            embeds = self.layer_norm(embeds)
        
        if 'ss8_seq' in self.args.structure_seq:
            ss_seq = batch['ss8_seq_input_ids']
            ss_embeds = self.ss_embedding(ss_seq)
            
            if 'foldseek_seq' in self.args.structure_seq:
                ss_embeds = self.cross_attention_ss(ss_embeds, embeds, embeds, attention_mask)
                embeds = ss_embeds + embeds
            else:
                ss_embeds = self.cross_attention_ss(ss_embeds, seq_embeds, seq_embeds, attention_mask)
                embeds = ss_embeds + seq_embeds
            embeds = self.layer_norm(embeds)
        
        if 'esm3_structure_seq' in self.args.structure_seq:
            esm3_structure_seq = batch['esm3_structure_seq_input_ids']
            esm3_structure_embeds = self.esm3_structure_embedding(esm3_structure_seq)
            
            if 'foldseek_seq' in self.args.structure_seq:
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, embeds, embeds, attention_mask)
                embeds = esm3_structure_embeds + embeds
            elif 'ss8_seq' in self.args.structure_seq:
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, embeds, embeds, attention_mask)
                embeds = esm3_structure_embeds + embeds
            else:
                esm3_structure_embeds = self.cross_attention_esm3_structure(esm3_structure_embeds, seq_embeds, seq_embeds, attention_mask)
                embeds = esm3_structure_embeds + seq_embeds
            embeds = self.layer_norm(embeds)
        
        # Process outputs
        features_to_use = embeds if self.args.structure_seq else seq_embeds
        
        if hasattr(self, 'pooling') and hasattr(self, 'projection'):
            pooled_features = self.pooling(features_to_use, attention_mask)
            logits = self.projection(pooled_features)
        else:
            logits = self.classifier(features_to_use, attention_mask)
        
        # Numerical stability check and improvement
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # For regression tasks, add slight regularization
        if self.args.problem_type == 'regression' and self.args.num_labels > 1:
            # Limit output range to avoid extreme values
            logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        return logits