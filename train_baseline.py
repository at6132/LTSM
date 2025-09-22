#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import time
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import pyarrow.parquet as pq
from tqdm import tqdm
import math

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(42)
torch.set_float32_matmul_precision('high')

class BinaryMoveDataset(Dataset):
    """Binary dataset for move prediction - predicts if market will move 0.35% within 5 bars."""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                 sequence_length: int = 120, expected_input_dim: int = None):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Enforce exact input dimensions - no automatic padding
        if expected_input_dim is not None and features.shape[1] != expected_input_dim:
            raise ValueError(f"Feature dimension mismatch: got {features.shape[1]}, expected {expected_input_dim}. "
                           f"This indicates a schema/scaler mismatch. Use load_features_from_checkpoint() for exact reproduction.")
        
        # Ensure we have enough data for sequences
        self.valid_length = len(self.features) - self.sequence_length + 1
        if self.valid_length <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}. Need at least {sequence_length} samples.")
        
    def __len__(self):
        return self.valid_length
    
    def __getitem__(self, idx):
        if idx >= self.valid_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.valid_length}")
            
        # Input sequence
        X = self.features[idx:idx + self.sequence_length]
        
        # Target index for current predictions
        target_idx = idx + self.sequence_length - 1
        
        # Binary target: 0 = no expected move, 1 = expected move
        move = int(self.targets['y_move'][target_idx])  # 0 or 1
        
        y = {
            'move': torch.LongTensor([move]),
        }
        
        return torch.FloatTensor(X), y

class SimpleDirectionalDataset(Dataset):
    """Simplified dataset for directional model - only direction prediction.
    FIXED: Uses all available samples properly."""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                 sequence_length: int = 120):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # FIXED: Ensure we have enough data for sequences
        self.valid_length = len(self.features) - self.sequence_length + 1
        if self.valid_length <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}. Need at least {sequence_length} samples.")
        
    def __len__(self):
        return self.valid_length
    
    def __getitem__(self, idx):
        if idx >= self.valid_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.valid_length}")
            
        # Input sequence
        X = self.features[idx:idx + self.sequence_length]
        
        # Target index for current predictions
        target_idx = idx + self.sequence_length - 1
        
        # Simplified target: direction with magnitude filter built-in
        direction = int(self.targets['y_direction'][target_idx])  # -1, 0, 1
        
        y = {
            'direction': torch.LongTensor([direction + 1]),  # Convert -1,0,1 to 0,1,2 for CrossEntropyLoss
        }
        
        return torch.FloatTensor(X), y

class MultiHeadAttention(nn.Module):
    """Enhanced multi-head attention for financial time series."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(out), attention

class AttentionPooling(nn.Module):
    """Attention-based pooling instead of using last timestep."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        attention_weights = self.attention(x)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        return pooled, attention_weights.squeeze(-1)

class TemporalConvolutions(nn.Module):
    """Multi-scale temporal convolutions for different time horizons."""
    
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        # Different kernel sizes for different time scales
        self.conv1 = nn.Conv1d(input_dim, d_model // 4, kernel_size=3, padding=1)   # Short-term
        self.conv2 = nn.Conv1d(input_dim, d_model // 4, kernel_size=7, padding=3)   # Medium-term
        self.conv3 = nn.Conv1d(input_dim, d_model // 4, kernel_size=15, padding=7)  # Long-term
        self.conv4 = nn.Conv1d(input_dim, d_model // 4, kernel_size=31, padding=15) # Very long-term
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim) -> (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply different convolutions
        conv1_out = self.activation(self.conv1(x))
        conv2_out = self.activation(self.conv2(x))
        conv3_out = self.activation(self.conv3(x))
        conv4_out = self.activation(self.conv4(x))
        
        # Concatenate outputs
        combined = torch.cat([conv1_out, conv2_out, conv3_out, conv4_out], dim=1)
        combined = self.batch_norm(combined)
        combined = self.dropout(combined)
        
        # Transpose back to (batch_size, seq_len, d_model)
        return combined.transpose(1, 2)

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_out, attention_weights = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attention_weights

class BinaryMoveModel(nn.Module):
    """Binary model for move prediction - predicts if market will move 0.35% within 5 bars."""
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Binary classification head (0=no move, 1=expected move)
        self.move_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary: 0 or 1
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Use last timestep for prediction
        last_hidden = x[:, -1, :]  # (batch_size, d_model)
        
        # Generate move prediction
        outputs = {
            'move': self.move_head(last_hidden),
            'attention_weights': attention_weights[-1]  # Last layer attention
        }
        
        return outputs

class SimpleDirectionalModel(nn.Module):
    """Simplified directional model focused on direction prediction only."""
    
    def __init__(self, input_dim: int, d_model: int = 256, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Direction classification head (0=short, 1=neutral, 2=long)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Use last timestep for prediction
        last_hidden = x[:, -1, :]  # (batch_size, d_model)
        
        # Generate direction prediction
        outputs = {
            'direction': self.direction_head(last_hidden),
            'attention_weights': attention_weights[-1]  # Last layer attention
        }
        
        return outputs

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    Supports scalar or per-class alpha weights.
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        # alpha can be None (no class weighting), a float, list[float], or torch.Tensor of shape [num_classes]
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha  # float | tensor | None
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, num_classes], targets: [batch]
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is None:
            alpha_t = 1.0
        elif isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != inputs.device:
                alpha_t = self.alpha.to(inputs.device)[targets]
            else:
                alpha_t = self.alpha[targets]
        else:
            alpha_t = float(self.alpha)
        
        loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for preventing overconfidence."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_preds = F.log_softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.smoothing) * targets_one_hot + self.smoothing / inputs.size(1)
        loss = (-targets_smooth * log_preds).sum(dim=1).mean()
        return loss

class HybridMoveLoss(nn.Module):
    """
    HYBRID loss that combines move-focused objectives with selectivity rewards.
    Finds the sweet spot between catching moves and being selective.
    """
    def __init__(self, move_recall_weight=3.0, move_precision_weight=2.0, 
                 selectivity_penalty=1.0, target_move_ratio=0.15):
        super().__init__()
        self.move_recall_weight = move_recall_weight      # How much we care about catching moves
        self.move_precision_weight = move_precision_weight # How much we care about precision
        self.selectivity_penalty = selectivity_penalty    # NEW: Penalty for being too aggressive
        self.target_move_ratio = target_move_ratio         # NEW: Target % of predictions that should be moves
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Hybrid loss: Move-focused + Selectivity rewards
        """
        logits = outputs['move']
        move_targets = targets['move'].long()
        if move_targets.dim() > 1:
            move_targets = move_targets.squeeze(-1)
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=1)
        move_probs = probs[:, 1]
        pred_labels = (move_probs > 0.5).long()
        
        # === PART 1: MOVE-FOCUSED OBJECTIVES (Natural) ===
        # Soft metrics for gradient flow
        soft_tp = move_probs * move_targets.float()
        soft_fp = move_probs * (1 - move_targets.float())
        soft_fn = (1 - move_probs) * move_targets.float()
        
        tp_sum = soft_tp.sum()
        fp_sum = soft_fp.sum()
        fn_sum = soft_fn.sum()
        
        # Move recall: % of actual moves caught (PRIMARY GOAL)
        move_recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        move_recall_loss = (1.0 - move_recall) * self.move_recall_weight
        
        # Move precision: % of move predictions that were correct (SECONDARY GOAL)
        move_precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        move_precision_loss = (1.0 - move_precision) * self.move_precision_weight
        
        # === PART 2: SELECTIVITY REWARD SYSTEM (Hybrid) ===
        # Calculate actual move prediction ratio
        batch_size = move_probs.size(0)
        actual_move_ratio = move_probs.mean()  # Average move probability across batch
        
        # Selectivity penalty: Penalize deviating from target ratio
        ratio_deviation = torch.abs(actual_move_ratio - self.target_move_ratio)
        selectivity_loss = ratio_deviation * self.selectivity_penalty
        
        # === PART 3: QUALITY BONUS (Reward System) ===
        # Bonus for high-confidence correct predictions
        if tp_sum > 0:  # Only if we made some correct move predictions
            # Reward high-confidence true positives
            high_conf_tp = (move_probs * move_targets.float() * (move_probs > 0.8).float()).sum()
            confidence_bonus = -0.1 * high_conf_tp  # Negative = reward
        else:
            confidence_bonus = torch.tensor(0.0, device=logits.device)  # Ensure it's a tensor
        
        # Total hybrid loss
        total_loss = move_recall_loss + move_precision_loss + selectivity_loss + confidence_bonus
        
        # Calculate hard metrics for reporting
        with torch.no_grad():
            hard_tp_mask = (pred_labels == 1) & (move_targets == 1)
            hard_fp_mask = (pred_labels == 1) & (move_targets == 0)
            hard_fn_mask = (pred_labels == 0) & (move_targets == 1)
            hard_tn_mask = (pred_labels == 0) & (move_targets == 0)
            
            hard_tp = hard_tp_mask.sum()
            hard_fp = hard_fp_mask.sum()
            hard_fn = hard_fn_mask.sum()
            hard_tn = hard_tn_mask.sum()
        
        return {
            'total': total_loss,
            'move_recall_loss': move_recall_loss,
            'move_precision_loss': move_precision_loss,
            'selectivity_loss': selectivity_loss,
            'confidence_bonus': confidence_bonus,
            'move_recall': move_recall,
            'move_precision': move_precision,
            'actual_move_ratio': actual_move_ratio,
            'target_move_ratio': torch.tensor(self.target_move_ratio),
            'precision': move_precision,  # For compatibility
            'recall': move_recall,  # For compatibility
            'f1_score': 2 * move_precision * move_recall / (move_precision + move_recall + 1e-8),
            'tp': hard_tp,
            'fp': hard_fp,
            'fn': hard_fn,
            'tn': hard_tn
        }

class MoveFocusedLoss(nn.Module):
    """
    Loss function that makes the model's PRIMARY GOAL catching moves.
    The model's 'reality' is that it's a move detector, not an overall classifier.
    """
    def __init__(self, move_recall_weight=10.0, move_precision_weight=1.0, no_move_accuracy_weight=0.1):
        super().__init__()
        self.move_recall_weight = move_recall_weight  # How much we care about catching moves
        self.move_precision_weight = move_precision_weight  # How much we care about false alarms
        self.no_move_accuracy_weight = no_move_accuracy_weight  # How much we care about no-move accuracy
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate move-focused loss where the PRIMARY objective is catching moves.
        """
        logits = outputs['move']
        move_targets = targets['move'].long()
        if move_targets.dim() > 1:
            move_targets = move_targets.squeeze(-1)
        
        # Get probabilities and predictions
        probs = F.softmax(logits, dim=1)
        move_probs = probs[:, 1]
        pred_labels = (move_probs > 0.5).long()
        
        # Use DIFFERENTIABLE operations for gradient flow
        # Instead of hard predictions, use soft probabilities
        
        # Soft TP: move_prob * move_target (high when both are high)
        soft_tp = move_probs * move_targets.float()
        
        # Soft FP: move_prob * (1 - move_target) (high when predicting move but no actual move)
        soft_fp = move_probs * (1 - move_targets.float())
        
        # Soft FN: (1 - move_prob) * move_target (high when missing actual move)
        soft_fn = (1 - move_probs) * move_targets.float()
        
        # Soft TN: (1 - move_prob) * (1 - move_target) (high when correctly ignoring no-move)
        soft_tn = (1 - move_probs) * (1 - move_targets.float())
        
        # Sum across batch for differentiable metrics
        tp_sum = soft_tp.sum()
        fp_sum = soft_fp.sum()
        fn_sum = soft_fn.sum()
        tn_sum = soft_tn.sum()
        
        # MOVE RECALL: What % of actual moves did we catch? (PRIMARY GOAL)
        move_recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        move_recall_loss = (1.0 - move_recall) * self.move_recall_weight
        
        # MOVE PRECISION: What % of our move predictions were correct? (Secondary goal)
        move_precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        move_precision_loss = (1.0 - move_precision) * self.move_precision_weight
        
        # NO-MOVE ACCURACY: What % of no-moves did we correctly ignore? (Tertiary goal)
        no_move_accuracy = tn_sum / (tn_sum + fp_sum + 1e-8)
        no_move_accuracy_loss = (1.0 - no_move_accuracy) * self.no_move_accuracy_weight
        
        # Total loss: Heavily weighted toward catching moves
        total_loss = move_recall_loss + move_precision_loss + no_move_accuracy_loss
        
        # Calculate hard metrics for reporting (detached from gradient)
        with torch.no_grad():
            hard_tp_mask = (pred_labels == 1) & (move_targets == 1)
            hard_fp_mask = (pred_labels == 1) & (move_targets == 0)
            hard_fn_mask = (pred_labels == 0) & (move_targets == 1)
            hard_tn_mask = (pred_labels == 0) & (move_targets == 0)
            
            hard_tp = hard_tp_mask.sum()
            hard_fp = hard_fp_mask.sum()
            hard_fn = hard_fn_mask.sum()
            hard_tn = hard_tn_mask.sum()
        
        return {
            'total': total_loss,
            'move_recall_loss': move_recall_loss,
            'move_precision_loss': move_precision_loss,
            'no_move_accuracy_loss': no_move_accuracy_loss,
            'move_recall': move_recall,
            'move_precision': move_precision,
            'no_move_accuracy': no_move_accuracy,
            'precision': move_precision,  # For compatibility
            'recall': move_recall,  # For compatibility
            'f1_score': 2 * move_precision * move_recall / (move_precision + move_recall + 1e-8),
            'tp': hard_tp,  # Use hard metrics for reporting
            'fp': hard_fp,
            'fn': hard_fn,
            'tn': hard_tn
        }

class RewardBasedLoss(nn.Module):
    """
    Reward-based loss function for binary move prediction.
    Rewards correct predictions and punishes incorrect ones with trading-specific logic.
    """
    def __init__(self, move_reward=2.0, false_positive_penalty=0.5, false_negative_penalty=1.0, 
                 entropy_weight=0.1, temperature=2.0):
        super().__init__()
        self.move_reward = move_reward  # 2 points for correct move prediction (scaled down!)
        self.false_positive_penalty = false_positive_penalty  # -0.5 for false positive
        self.false_negative_penalty = false_negative_penalty  # -1 for false negative 
        self.entropy_weight = entropy_weight  # Encourage uncertainty
        self.temperature = temperature  # Temperature scaling to reduce overconfidence
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate reward-based loss.
        
        Args:
            outputs: Model outputs with 'move' key containing logits [batch_size, 2]
            targets: True labels with 'move' key [batch_size] (0=no move, 1=move)
            
        Returns:
            Dictionary with loss components
        """
        logits = outputs['move']
        move_targets = targets['move'].long()
        # Only squeeze if there are extra dimensions beyond batch
        if move_targets.dim() > 1:
            move_targets = move_targets.squeeze(-1)
        
        # Apply temperature scaling to reduce overconfidence
        scaled_logits = logits / self.temperature
        
        # Convert logits to probabilities
        probs = F.softmax(scaled_logits, dim=1)
        move_probs = probs[:, 1]  # Probability of move
        
        # Get predictions (0 or 1) - use probability threshold to avoid argmax bias
        pred_labels = (move_probs > 0.5).long()
        
        # Go back to reward-based approach but with CORRECT gradient direction
        # The key insight: we want to MAXIMIZE rewards, so loss = -rewards
        
        # Calculate rewards using LOGITS directly (not probabilities) for correct gradients
        move_logits_scaled = scaled_logits[:, 1]  # Scaled move logits
        no_move_logits_scaled = scaled_logits[:, 0]  # Scaled no-move logits
        
        # Direct loss formulation that pushes gradients in correct direction
        # We want to MINIMIZE this loss, which means:
        # - INCREASE move_logits when target=1 (to get reward)
        # - DECREASE no_move_logits when target=1 (to avoid penalty)
        # - DECREASE move_logits when target=0 (to avoid penalty)
        
        # For true moves (target=1): we want high move_logits, low no_move_logits
        move_loss = -move_logits_scaled * move_targets.float() * self.move_reward  # negative = reward
        no_move_penalty = no_move_logits_scaled * move_targets.float() * self.false_negative_penalty  # positive = penalty
        
        # For true no-moves (target=0): we want low move_logits
        false_positive_penalty = move_logits_scaled * (1 - move_targets.float()) * self.false_positive_penalty  # positive = penalty
        
        # Total loss to minimize
        total_loss = move_loss + no_move_penalty + false_positive_penalty
        
        # Add entropy regularization (encourage uncertainty)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        entropy_bonus = -self.entropy_weight * entropy  # negative because we want to minimize loss
        
        # Final loss with clipping to prevent explosion
        loss = total_loss.mean() + entropy_bonus.mean()
        
        # Clip loss to prevent numerical instability
        loss = torch.clamp(loss, min=-10.0, max=10.0)
        
        # Calculate move probabilities for metrics
        move_probs = probs[:, 1]
        
        # Calculate metrics using hard predictions for evaluation
        tp_mask = (pred_labels == 1) & (move_targets == 1)
        fp_mask = (pred_labels == 1) & (move_targets == 0)
        fn_mask = (pred_labels == 0) & (move_targets == 1)
        tn_mask = (pred_labels == 0) & (move_targets == 0)
        
        tp = tp_mask.sum()
        fp = fp_mask.sum()
        fn = fn_mask.sum()
        tn = tn_mask.sum()
        
        # Calculate precision, recall, f1 as tensors
        precision = tp / (tp + fp + 1e-8)  # Add small epsilon to avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'total': loss,
            'move_reward': torch.tensor(self.move_reward, device=loss.device),
            'false_positive_penalty': torch.tensor(self.false_positive_penalty, device=loss.device),
            'false_negative_penalty': torch.tensor(self.false_negative_penalty, device=loss.device),
            'entropy_bonus': entropy_bonus.mean(),
            'avg_entropy': entropy.mean(),
            'avg_confidence': torch.max(probs, dim=1)[0].mean(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

class BinaryMoveLoss(nn.Module):
    """Binary loss function for move prediction."""
    
    def __init__(self, focal_alpha: Optional[torch.Tensor] = None, focal_gamma: float = 1.5,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing) if label_smoothing > 0 else None
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        logits = outputs['move']
        move_targets = targets['move'].long().squeeze()
        
        # Primary focal loss
        focal = self.focal_loss(logits, move_targets)
        total = focal
        losses['move_focal'] = focal
        
        # Optional label smoothing (lighter weight to prevent conflicts)
        if self.label_smooth_loss is not None:
            ls = self.label_smooth_loss(logits, move_targets)
            total = total + 0.1 * ls  # Very small contribution
            losses['move_label_smooth'] = ls
        
        losses['total'] = total
        return losses

class SimpleDirectionalLoss(nn.Module):
    """Simplified loss function focusing on stable training."""
    
    def __init__(self, focal_alpha: Optional[torch.Tensor] = None, focal_gamma: float = 1.5,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smooth_loss = LabelSmoothingCrossEntropy(smoothing=label_smoothing) if label_smoothing > 0 else None
    
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        losses = {}
        logits = outputs['direction']
        direction_targets = targets['direction'].long().squeeze()
        
        # Primary focal loss
        focal = self.focal_loss(logits, direction_targets)
        total = focal
        losses['direction_focal'] = focal
        
        # Optional label smoothing (lighter weight to prevent conflicts)
        if self.label_smooth_loss is not None:
            ls = self.label_smooth_loss(logits, direction_targets)
            total = total + 0.1 * ls  # Very small contribution
            losses['direction_label_smooth'] = ls
        
        losses['total'] = total
        return losses

def create_binary_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create binary targets for move prediction."""
    
    targets = {}
    
    # Reset index to ensure continuous indexing
    df_reset = df.reset_index(drop=True)
    
    # Binary move prediction (0 = no move, 1 = expected move)
    targets['y_move'] = df_reset['y_move'].astype(int).values
    
    return targets

def create_directional_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create simplified directional targets with magnitude filtering."""
    
    targets = {}
    
    # Reset index to ensure continuous indexing
    df_reset = df.reset_index(drop=True)
    
    # Direction with magnitude filter built-in (-1, 0, 1)
    targets['y_direction'] = df_reset['y_direction'].astype(int).values
    
    return targets

def curriculum_learning_schedule(epoch: int, total_epochs: int) -> float:
    """Curriculum learning schedule - start with easy examples, add complexity."""
    # Start with 20% of data, gradually increase to 100%
    progress = epoch / total_epochs
    if progress < 0.2:
        return 0.2  # Start with 20% of data
    elif progress < 0.5:
        return 0.2 + 0.6 * (progress - 0.2) / 0.3  # Linear increase to 80%
    else:
        return 0.8 + 0.2 * (progress - 0.5) / 0.5  # Linear increase to 100%

def prepare_features(df: pd.DataFrame, feature_cols: List[str], scaler=None, fit_scaler=False) -> np.ndarray:
    """Prepare and scale features with proper standardization to handle data drift."""
    from sklearn.preprocessing import StandardScaler, RobustScaler
    
    df_reset = df.reset_index(drop=True)
    feature_data = df_reset[feature_cols].copy()
    
    # Handle missing values
    feature_data = feature_data.ffill().fillna(0)
    
    # Volume features need different scaling than price features
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'signed_vol', 'dCVD', 'CVD']
    
    # Pre-process features
    for col in feature_data.columns:
        if col in volume_features:
            # Scale volume features to millions
            feature_data[col] = feature_data[col] / 1e6
        elif feature_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Clip outliers for other features
            p1, p99 = np.percentile(feature_data[col], [1, 99])
            feature_data[col] = feature_data[col].clip(p1, p99)
    
    # Apply robust standardization to handle data drift
    if scaler is None:
        # Use RobustScaler for better handling of outliers and drift
        scaler = RobustScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
    else:
        if fit_scaler:
            feature_data_scaled = scaler.fit_transform(feature_data)
        else:
            feature_data_scaled = scaler.transform(feature_data)
    
    return feature_data_scaled.astype(np.float32), scaler

def load_features_from_checkpoint(df: pd.DataFrame, checkpoint_path: str, data_split: str = 'test'):
    """Load and prepare features exactly as they were during training.
    
    Args:
        df: Full dataframe
        checkpoint_path: Path to saved model checkpoint
        data_split: Which split to prepare ('train', 'val', 'test')
    
    Returns:
        features: Prepared features exactly as during training
        split_data: The actual data split used
    """
    # Load checkpoint to get exact training configuration
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract saved training components
    args = checkpoint['args']
    saved_feature_cols = checkpoint['feature_cols']
    robust_scaler = checkpoint['robust_scaler']
    standard_scaler = checkpoint['standard_scaler']
    
    print(f"🔧 Loading features from checkpoint:")
    print(f"   Saved feature columns: {len(saved_feature_cols)}")
    print(f"   Expected input_dim: {len(saved_feature_cols)}")
    
    # Create exact same walk-forward split as training
    splits = walk_forward_split(df, args, args.train_months, args.val_months, args.test_months, 
                               args.stride_days, args.max_splits)
    if len(splits) == 0:
        raise ValueError("No splits created - check data range")
    
    split = splits[0]  # Use same split as training
    
    # Get the requested data split
    if data_split == 'train':
        split_data = split['train']
    elif data_split == 'val':
        split_data = split['val']
    elif data_split == 'test':
        split_data = split['test']
    else:
        raise ValueError(f"Invalid data_split: {data_split}")
    
    print(f"   {data_split.capitalize()} period: {split[f'{data_split}_dates'][0]} to {split[f'{data_split}_dates'][-1]}")
    print(f"   {data_split.capitalize()} samples: {len(split_data):,}")
    
    # Prepare features using EXACT saved feature columns and scalers
    df_reset = split_data.reset_index(drop=True)
    
    # Verify all required columns exist
    missing_cols = [col for col in saved_feature_cols if col not in df_reset.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    
    # Extract features in EXACT same order as training
    feature_data = df_reset[saved_feature_cols].copy()
    
    # Apply EXACT same preprocessing as training
    feature_data = feature_data.ffill().fillna(0)
    
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'signed_vol', 'dCVD', 'CVD']
    
    for col in feature_data.columns:
        if col in volume_features:
            feature_data[col] = feature_data[col] / 1e6
        elif feature_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            p1, p99 = np.percentile(feature_data[col], [1, 99])
            feature_data[col] = feature_data[col].clip(p1, p99)
    
    # Apply EXACT saved RobustScaler (no fitting, just transform)
    features_robust_scaled = robust_scaler.transform(feature_data)
    
    # Apply EXACT saved StandardScaler (no fitting, just transform)  
    features_final = standard_scaler.transform(features_robust_scaled)
    
    print(f"   ✅ Features prepared with saved scalers")
    print(f"   Final feature shape: {features_final.shape}")
    
    return features_final.astype(np.float32), split_data

def calculate_advanced_metrics(predictions: Dict[str, np.ndarray], 
                              targets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate performance metrics for directional model only."""
    
    metrics = {}
    
    # Only calculate metrics for available predictions
    if 'direction' in predictions and 'direction' in targets:
        # Direction classification accuracy
        direction_pred = np.argmax(predictions['direction'], axis=1)
        direction_true = targets['direction'].astype(int).squeeze()
        metrics['direction_accuracy'] = accuracy_score(direction_true, direction_pred)
        
        # Class distribution analysis
        unique, counts = np.unique(direction_true, return_counts=True)
        metrics['class_distribution'] = dict(zip(unique, counts))
        
        # Per-class accuracy
        for class_id in [0, 1, 2]:  # short, neutral, long
            mask = direction_true == class_id
            if mask.sum() > 0:
                class_acc = accuracy_score(direction_true[mask], direction_pred[mask])
                metrics[f'class_{class_id}_accuracy'] = class_acc
    
    return metrics

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: optim.Optimizer,
                criterion: SimpleDirectionalLoss, device: torch.device, 
                scaler: GradScaler = None, clip_grad: float = 1.0) -> Dict[str, float]:
    """Train for one epoch with simplified approach."""
    model.train()
    total_losses = {}
    num_batches = 0
    
    for batch_features, batch_targets in tqdm(dataloader, desc="Training"):
        batch_features = batch_features.to(device, non_blocking=True)
        batch_targets = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with autocast():
                outputs = model(batch_features)
                losses = criterion(outputs, batch_targets)
            
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(batch_features)
            losses = criterion(outputs, batch_targets)
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
        
        # Accumulate losses
        for key, loss in losses.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += loss.item()
        num_batches += 1
    
    return {k: v / num_batches for k, v in total_losses.items()}

def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: SimpleDirectionalLoss,
                   device: torch.device, scaler: GradScaler = None) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, float]]:
    """Evaluate the model and return losses, predictions, and metrics."""
    model.eval()
    total_losses = {}
    num_batches = 0
    
    all_predictions = {}
    all_targets = {}
    
    with torch.no_grad():
        for batch_features, batch_targets in tqdm(dataloader, desc="Evaluating"):
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = {k: v.to(device, non_blocking=True) for k, v in batch_targets.items()}
            
            if scaler is not None:
                with autocast():
                    outputs = model(batch_features)
                    losses = criterion(outputs, batch_targets)
            else:
                outputs = model(batch_features)
                losses = criterion(outputs, batch_targets)
            
            # Accumulate losses
            for key, loss in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += loss.item()
            num_batches += 1
            
            # Store predictions and targets
            for key, output in outputs.items():
                if key == 'attention_weights':  # Skip attention weights
                    continue
                if key not in all_predictions:
                    all_predictions[key] = []
                all_predictions[key].append(output.cpu().numpy())
            
            for key, target in batch_targets.items():
                if key not in all_targets:
                    all_targets[key] = []
                all_targets[key].append(target.cpu().numpy())
    
    # Concatenate all predictions and targets
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
    for key in all_targets:
        all_targets[key] = np.concatenate(all_targets[key], axis=0)
    
    # Calculate metrics based on mode
    if 'move' in all_predictions:
        metrics = calculate_binary_metrics(all_predictions, all_targets)
    else:
        metrics = calculate_directional_metrics(all_predictions, all_targets)
    
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses, all_predictions, metrics

def calculate_binary_metrics(predictions: Dict[str, np.ndarray], 
                           targets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate metrics for binary move prediction model."""
    
    metrics = {}
    
    # Binary classification accuracy
    move_pred = np.argmax(predictions['move'], axis=1)
    move_true = targets['move'].astype(int).squeeze()
    metrics['move_accuracy'] = accuracy_score(move_true, move_pred)
    
    # Class distribution (true) and predicted distribution
    true_counts = np.bincount(move_true, minlength=2)
    pred_counts = np.bincount(move_pred, minlength=2)
    
    metrics['true_distribution'] = {
        'no_move': true_counts[0],
        'expected_move': true_counts[1]
    }
    metrics['predicted_distribution'] = {
        'no_move': pred_counts[0],
        'expected_move': pred_counts[1]
    }
    
    # Calculate precision, recall, F1 for expected moves (class 1)
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    if len(np.unique(move_true)) > 1:  # Both classes present
        metrics['precision'] = precision_score(move_true, move_pred, average='binary')
        metrics['recall'] = recall_score(move_true, move_pred, average='binary')
        metrics['f1_score'] = f1_score(move_true, move_pred, average='binary')
    else:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1_score'] = 0.0
    
    return metrics

def calculate_directional_metrics(predictions: Dict[str, np.ndarray], 
                                 targets: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Calculate metrics for directional model with bias analysis."""
    
    metrics = {}
    
    # Direction classification accuracy
    direction_pred = np.argmax(predictions['direction'], axis=1)
    direction_true = targets['direction'].astype(int).squeeze()
    metrics['direction_accuracy'] = accuracy_score(direction_true, direction_pred)
    
    # Class distribution (true) and predicted distribution
    unique, counts = np.unique(direction_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique, counts))
    pred_unique, pred_counts = np.unique(direction_pred, return_counts=True)
    metrics['predicted_distribution'] = dict(zip(pred_unique, pred_counts))
    
    # FIXED: Calculate bias metrics
    true_dist = np.bincount(direction_true, minlength=3) / len(direction_true)
    pred_dist = np.bincount(direction_pred, minlength=3) / len(direction_pred)
    
    bias_metrics = {}
    class_names = ['Short', 'Neutral', 'Long']
    for i, name in enumerate(class_names):
        bias_metrics[f'{name.lower()}_bias'] = pred_dist[i] - true_dist[i]
    
    metrics.update(bias_metrics)
    
    # Per-class accuracy
    for class_id in [0, 1, 2]:  # short, neutral, long
        mask = direction_true == class_id
        if mask.sum() > 0:
            class_acc = accuracy_score(direction_true[mask], direction_pred[mask])
            metrics[f'class_{class_id}_accuracy'] = class_acc
    
    return metrics

def walk_forward_split(df: pd.DataFrame, args, train_months: int = 12, 
                      val_months: int = 3, test_months: int = 3, 
                      stride_days: int = 30, max_splits: int = None) -> List[Dict]:
    """Create walk-forward validation splits with balanced validation sets.
    FIXED: Ensures validation sets have reasonable class distribution."""
    df['date'] = pd.to_datetime(df['ts']).dt.date
    unique_dates = sorted(df['date'].unique())
    
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"Total days: {len(unique_dates)}")
    print(f"Required days per split: {train_months * 30.4 + val_months * 30.4 + test_months * 30.4}")
    
    train_days = int(train_months * 30.4)
    val_days = int(val_months * 30.4)
    test_days = int(test_months * 30.4)
    
    print(f"Train days: {train_days}, Val days: {val_days}, Test days: {test_days}")
    print(f"Total required: {train_days + val_days + test_days}")
    
    splits = []
    
    # FIXED: Start from the END (most recent data) instead of beginning (old data)
    # For monthly retraining, we want the MOST RECENT periods
    total_required_days = train_days + val_days + test_days
    
    if len(unique_dates) < total_required_days:
        raise ValueError(f"Not enough data: need {total_required_days} days, have {len(unique_dates)}")
    
    # Calculate starting positions from the END of data
    if max_splits is None:
        max_splits = 1  # Default to 1 split for recent data
    
    k = 0
    # Start from most recent data and work backwards
    for split_num in range(max_splits):
        # Calculate end position (working backwards from latest data)
        end_idx = len(unique_dates) - (split_num * stride_days)
        start_idx = end_idx - total_required_days
        
        if start_idx < 0:
            break  # Not enough historical data for this split
            
        # Ensure we have enough data
        if start_idx + total_required_days > len(unique_dates):
            break
        train_end = start_idx + train_days
        val_end = train_end + val_days
        test_end = val_end + test_days
        
        train_dates = unique_dates[start_idx:train_end]
        val_dates = unique_dates[train_end:val_end]
        test_dates = unique_dates[val_end:test_end]
        
        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        test_mask = df['date'].isin(test_dates)
        
        train_data = df[train_mask].copy()
        val_data = df[val_mask].copy()
        test_data = df[test_mask].copy()
        
        # Print class distributions for each split
        if args.mode == "binary":
            train_dist = train_data['y_move'].value_counts(normalize=True).sort_index() * 100
            val_dist = val_data['y_move'].value_counts(normalize=True).sort_index() * 100
            test_dist = test_data['y_move'].value_counts(normalize=True).sort_index() * 100
            
            print(f"  Train: No Move={train_dist.get(0, 0):.1f}%, Expected Move={train_dist.get(1, 0):.1f}%")
            print(f"  Val:   No Move={val_dist.get(0, 0):.1f}%, Expected Move={val_dist.get(1, 0):.1f}%")
            print(f"  Test:  No Move={test_dist.get(0, 0):.1f}%, Expected Move={test_dist.get(1, 0):.1f}%")
        else:
            # FIXED: Check validation set balance and adjust if needed
            val_neutral_pct = (val_data['y_direction'] == 0).mean() * 100
            
            if val_neutral_pct < 0.5:  # Less than 0.5% neutral samples
                print(f"⚠️  Split {k+1}: Validation set has only {val_neutral_pct:.2f}% neutral samples")
                print(f"   Extending validation period to improve balance...")
                
                # Extend validation period by 30 days to get more neutral samples
                extended_val_end = min(val_end + 30, len(unique_dates) - test_days)
                if extended_val_end > val_end:
                    val_dates_extended = unique_dates[train_end:extended_val_end]
                    val_data = df[df['date'].isin(val_dates_extended)].copy()
                    
                    # Adjust test period accordingly
                    test_start = extended_val_end
                    test_end_new = min(test_start + test_days, len(unique_dates))
                    test_dates_new = unique_dates[test_start:test_end_new]
                    test_data = df[df['date'].isin(test_dates_new)].copy()
                    
                    val_neutral_pct_new = (val_data['y_direction'] == 0).mean() * 100
                    print(f"   Extended validation: {val_neutral_pct:.2f}% → {val_neutral_pct_new:.2f}% neutral samples")
            
            train_dist = train_data['y_direction'].value_counts(normalize=True).sort_index() * 100
            val_dist = val_data['y_direction'].value_counts(normalize=True).sort_index() * 100
            test_dist = test_data['y_direction'].value_counts(normalize=True).sort_index() * 100
        
        splits.append({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates
        })
        
        print(f"Split {k+1}: Train {train_dates[0]} to {train_dates[-1]}, "
              f"Val {val_dates[0]} to {val_dates[-1]}, "
              f"Test {test_dates[0]} to {test_dates[-1]}")
        
        if args.mode == "directional":
            print(f"  Train: Short={train_dist.get(-1, 0):.1f}%, Neutral={train_dist.get(0, 0):.1f}%, Long={train_dist.get(1, 0):.1f}%")
            print(f"  Val:   Short={val_dist.get(-1, 0):.1f}%, Neutral={val_dist.get(0, 0):.1f}%, Long={val_dist.get(1, 0):.1f}%")
            print(f"  Test:  Short={test_dist.get(-1, 0):.1f}%, Neutral={test_dist.get(0, 0):.1f}%, Long={test_dist.get(1, 0):.1f}%")
        
        k += 1
        # Note: We're already iterating through max_splits in the for loop above
    
    return splits

def compute_class_balanced_weights(class_counts: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Compute class-balanced weights using the effective number of samples.
    Args:
        class_counts: array shape [num_classes]
        beta: hyperparameter in [0,1). Higher -> smoother weights.
    Returns:
        weights: array shape [num_classes], normalized to mean 1.0
    """
    class_counts = class_counts.astype(np.float64)
    eps = 1e-12
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.maximum(effective_num, eps)
    # Normalize to mean 1.0 for stable loss scale
    weights = weights / (np.mean(weights) + eps)
    return weights

def main():
    parser = argparse.ArgumentParser(description="Train trading model")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--mode", type=str, default="binary", 
                        choices=["binary", "directional"],
                        help="Model mode: binary (move prediction) or directional (3-class)")
    parser.add_argument("--sequence_length", type=int, default=120)
    parser.add_argument("--prediction_horizon", type=int, default=10)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)  # Reduced due to model complexity
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train_months", type=int, default=12)
    parser.add_argument("--val_months", type=int, default=3)
    parser.add_argument("--test_months", type=int, default=3)
    parser.add_argument("--stride_days", type=int, default=30)
    parser.add_argument("--max_splits", type=int, default=None)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    print(f"Loading data for {args.symbol} {args.interval}...")
    
    # Load features and labels with error handling
    try:
        features_df = pq.read_table(f"features/features_{args.symbol}_{args.interval}.parquet").to_pandas()
        labels_df = pq.read_table(f"features/labels_{args.symbol}_{args.interval}.parquet").to_pandas()
    except FileNotFoundError as e:
        print(f"ERROR: Required files not found: {e}")
        print("Please run build_trading_labels.py first to generate the labels file.")
        return
    
    # Merge features and labels
    df = features_df.merge(labels_df, on='ts', how='inner')
    print(f"Loaded {len(df)} samples")
    
    # Validate data
    if len(df) == 0:
        print("ERROR: No data after merging features and labels!")
        return
    
    # Check for required columns based on mode
    if args.mode == "binary":
        required_cols = ['y_move']
        exclude_cols = ['ts', 'y_hit', 'y_tth', 'y_direction', 'y_move']
    else:
        required_cols = ['y_direction']
        exclude_cols = ['ts', 'y_hit', 'y_tth', 'y_direction', 'y_move']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return
    
    # Remove duplicate features identified in analysis
    duplicate_features = ['high', 'low', 'close', 'tot_vol', 'atr_pct', 'dCVD']
    
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col not in duplicate_features]
    
    print(f"Using {len(feature_cols)} features (removed {len(duplicate_features)} duplicates)")
    print(f"Removed duplicates: {duplicate_features}")
    
    # Create walk-forward splits
    splits = walk_forward_split(df, args, args.train_months, args.val_months, args.test_months, 
                               args.stride_days, args.max_splits)
    print(f"Created {len(splits)} walk-forward splits")
    
    if len(splits) == 0:
        print("ERROR: No walk-forward splits created!")
        print("Try reducing train_months, val_months, or test_months")
        print("Or check if your data has enough date range")
        return
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    all_results = []
    
    for split_idx, split in enumerate(splits):
        print(f"\n=== Split {split_idx + 1}/{len(splits)} ===")
        print(f"Train: {split['train_dates'][0]} to {split['train_dates'][-1]}")
        print(f"Val: {split['val_dates'][0]} to {split['val_dates'][-1]}")
        print(f"Test: {split['test_dates'][0]} to {split['test_dates'][-1]}")
        
        # Prepare features
        # Prepare features with proper scaling
        train_features, feature_scaler = prepare_features(split['train'], feature_cols, scaler=None, fit_scaler=True)
        val_features, _ = prepare_features(split['val'], feature_cols, scaler=feature_scaler, fit_scaler=False)
        test_features, _ = prepare_features(split['test'], feature_cols, scaler=feature_scaler, fit_scaler=False)
        
        # Create targets based on mode
        if args.mode == "binary":
            train_targets = create_binary_targets(split['train'])
            val_targets = create_binary_targets(split['val'])
            test_targets = create_binary_targets(split['test'])
        else:
            train_targets = create_directional_targets(split['train'])
            val_targets = create_directional_targets(split['val'])
            test_targets = create_directional_targets(split['test'])
        
        # Scale features
        standard_scaler = StandardScaler()
        train_features = standard_scaler.fit_transform(train_features)
        val_features = standard_scaler.transform(val_features)
        test_features = standard_scaler.transform(test_features)
        
        # Create datasets based on mode
        if args.mode == "binary":
            train_dataset = BinaryMoveDataset(train_features, train_targets, 
                                           args.sequence_length)
            val_dataset = BinaryMoveDataset(val_features, val_targets, 
                                         args.sequence_length)
            test_dataset = BinaryMoveDataset(test_features, test_targets, 
                                          args.sequence_length)
        else:
            train_dataset = SimpleDirectionalDataset(train_features, train_targets, 
                                                   args.sequence_length)
            val_dataset = SimpleDirectionalDataset(val_features, val_targets, 
                                                 args.sequence_length)
            test_dataset = SimpleDirectionalDataset(test_features, test_targets, 
                                                  args.sequence_length)
        
        # FIXED: Compute class weights based on ACTUAL samples used by dataset
        seq_len = args.sequence_length
        num_samples = len(train_dataset)
        
        # Count classes in the actual samples used by the dataset
        if args.mode == "binary":
            used_targets = train_targets['y_move'][seq_len-1:seq_len-1+num_samples]
            used_classes = used_targets.astype(int)  # 0 or 1
            class_counts = np.bincount(used_classes, minlength=2)
            
            print(f"Dataset samples: {num_samples:,}")
            print(f"Used class counts: {class_counts.tolist()}")
            print(f"Used class distribution: No Move={class_counts[0]/num_samples*100:.1f}%, Expected Move={class_counts[1]/num_samples*100:.1f}%")
            
            # Use balanced weights for binary classification
            class_weights = np.array([1.0, 1.0], dtype=np.float32)
            print(f"Using fixed class weights: {class_weights.tolist()}")
            
            # Build per-sample weights for sequence dataset
            sample_weights = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_samples):
                cls_id = int(used_targets[i])  # 0 or 1
                sample_weights[i] = class_weights[cls_id]
        else:
            used_targets = train_targets['y_direction'][seq_len-1:seq_len-1+num_samples]
            used_classes = (used_targets.astype(int) + 1)  # Convert -1,0,1 to 0,1,2
            class_counts = np.bincount(used_classes, minlength=3)
            
            print(f"Dataset samples: {num_samples:,}")
            print(f"Used class counts: {class_counts.tolist()}")
            print(f"Used class distribution: Short={class_counts[0]/num_samples*100:.1f}%, Neutral={class_counts[1]/num_samples*100:.1f}%, Long={class_counts[2]/num_samples*100:.1f}%")
            
            # Use balanced weights to encourage all 3 classes: [Short, Neutral, Long]
            class_weights = np.array([2.0, 1.0, 2.0], dtype=np.float32)
            print(f"Using fixed class weights: {class_weights.tolist()}")
            
            # Build per-sample weights for sequence dataset
            sample_weights = np.zeros(num_samples, dtype=np.float32)
            for i in range(num_samples):
                cls_id = int(used_targets[i]) + 1  # Use the actual used targets
                sample_weights[i] = class_weights[cls_id]
        
        # FIXED: Use balanced sampling instead of weighted sampling to prevent extreme bias
        # WeightedRandomSampler can create extreme bias, so use stratified sampling
        sampler = None  # Use regular sampling with class weights in loss function instead
        
        # FIXED: Create data loaders with proper sampling
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                sampler=sampler,  # None for regular sampling
                                num_workers=min(8, os.cpu_count()//2), pin_memory=True,
                                persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=min(8, os.cpu_count()//2), pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=min(8, os.cpu_count()//2), pin_memory=True,
                               persistent_workers=True, prefetch_factor=2)
        
        print(f"Train samples: {len(train_dataset):,}")
        print(f"Val samples: {len(val_dataset):,}")
        print(f"Test samples: {len(test_dataset):,}")
        
        # FIXED: Validate dataset creation
        print(f"\nDataset Validation:")
        print(f"  Original train data: {len(split['train']):,} samples")
        print(f"  Dataset uses: {len(train_dataset):,} samples")
        print(f"  Lost samples: {len(split['train']) - len(train_dataset):,}")
        print(f"  Loss percentage: {(len(split['train']) - len(train_dataset)) / len(split['train']) * 100:.2f}%")
        
        # Create model based on mode
        if args.mode == "binary":
            model = BinaryMoveModel(
                input_dim=len(feature_cols),
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                d_ff=args.d_ff,
                dropout=args.dropout
            ).to(device)
        else:
            model = SimpleDirectionalModel(
                input_dim=len(feature_cols),
                d_model=args.d_model,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                d_ff=args.d_ff,
                dropout=args.dropout
            ).to(device)
        
        # Initialize model weights properly to prevent extreme bias
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        print("Model weights initialized with Xavier uniform")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        # Check initial predictions before training
        model.eval()
        with torch.no_grad():
            sample_batch = next(iter(train_loader))
            sample_features = sample_batch[0][:5].to(device)  # First 5 samples
            initial_outputs = model(sample_features)
            if args.mode == "binary":
                initial_probs = torch.softmax(initial_outputs['move'], dim=1)
                print(f"Initial prediction probabilities (first 5 samples):")
                for i in range(5):
                    probs = initial_probs[i].cpu().numpy()
                    print(f"  Sample {i}: No Move={probs[0]:.3f}, Expected Move={probs[1]:.3f}")
            else:
                initial_probs = torch.softmax(initial_outputs['direction'], dim=1)
                print(f"Initial prediction probabilities (first 5 samples):")
                for i in range(5):
                    probs = initial_probs[i].cpu().numpy()
                    print(f"  Sample {i}: Short={probs[0]:.3f}, Neutral={probs[1]:.3f}, Long={probs[2]:.3f}")
        model.train()
        
        # Create loss function based on mode - MOVE-FOCUSED OBJECTIVE
        if args.mode == "binary":
            print("🎯 Using MOVE-FOCUSED loss: Model's goal is to catch moves, not overall accuracy")
            
            criterion = HybridMoveLoss(
                move_recall_weight=3.0,      # Want to catch moves (reduced from 10.0)
                move_precision_weight=2.0,   # Care about precision too (increased from 1.0)
                selectivity_penalty=1.0,     # NEW: Penalize predicting too many moves
                target_move_ratio=0.15       # NEW: Ideal move prediction rate (15% instead of 100%)
            )
        else:
            criterion = SimpleDirectionalLoss(
                focal_alpha=torch.tensor(class_weights, dtype=torch.float32),
                focal_gamma=1.0,  # Standard gamma
                label_smoothing=0.05  # Light smoothing
            )
        
        # FIXED: Create optimizer with stronger regularization
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, 
                               betas=(0.9, 0.95), eps=1e-8)
        
        # FIXED: Add stronger gradient clipping to prevent extreme updates
        max_grad_norm = 0.5
        
        # FIXED: Learning rate scheduler with more aggressive decay
        def lr_lambda(step):
            if step < args.warmup_epochs:
                return step / args.warmup_epochs
            return 0.7 ** ((step - args.warmup_epochs) // 15)  # More frequent decay
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Mixed precision scaler
        scaler = GradScaler() if (device.type == 'cuda' and args.mixed_precision) else None
        
        # CUDA memory management
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # FIXED: Training loop with better early stopping
        best_val_loss = float('inf')
        best_direction_acc = 0.0
        patience = 20  # Increased patience for better convergence
        patience_counter = 0
        
        print(f"Starting training for split {split_idx + 1}...")
        
        for epoch in range(args.epochs):
            # Training
            train_losses = train_epoch(model, train_loader, optimizer, criterion, 
                                     device, scaler, max_grad_norm)
            
            # Validation
            val_losses, val_predictions, val_metrics = evaluate_model(model, val_loader, 
                                                                    criterion, device, scaler)
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # FIXED: Print progress with bias analysis
            if epoch % 5 == 0 or epoch < 10:
                print(f"Epoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"  Val Loss: {val_losses['total']:.4f}")
                
                if args.mode == "binary":
                    print(f"  Val Move Acc: {val_metrics['move_accuracy']:.4f}")
                    print(f"  Precision: {val_metrics.get('precision', 0):.4f}")
                    print(f"  Recall: {val_metrics.get('recall', 0):.4f}")
                    print(f"  F1 Score: {val_metrics.get('f1_score', 0):.4f}")
                    
                    true_dist = val_metrics.get('true_distribution', {})
                    pred_dist = val_metrics.get('predicted_distribution', {})
                    print(f"  True Distribution: No Move={true_dist.get('no_move', 0)}, Expected Move={true_dist.get('expected_move', 0)}")
                    print(f"  Predicted Distribution: No Move={pred_dist.get('no_move', 0)}, Expected Move={pred_dist.get('expected_move', 0)}")
                else:
                    print(f"  Val Direction Acc: {val_metrics['direction_accuracy']:.4f}")
                    print(f"  Class Distribution: {val_metrics.get('class_distribution', {})}")
                    pred_dist = val_metrics.get('predicted_distribution', {})
                    # Order requested: 1, 0, 2
                    pred_counts = {k: pred_dist.get(k, 0) for k in [1, 0, 2]}
                    total_preds = max(1, sum(pred_dist.values()))
                    pred_pct = {k: round(100.0 * pred_counts[k] / total_preds, 2) for k in pred_counts}
                    print(f"  Predicted Distribution (counts) [1,0,2]: {pred_counts}")
                    print(f"  Predicted Distribution (pct)    [1,0,2]: {pred_pct}")
                    
                    # FIXED: Show bias metrics
                    print(f"  Bias Analysis:")
                    print(f"    Short Bias:  {val_metrics.get('short_bias', 0):+.3f}")
                    print(f"    Neutral Bias: {val_metrics.get('neutral_bias', 0):+.3f}")
                    print(f"    Long Bias:   {val_metrics.get('long_bias', 0):+.3f}")
                
                # Debug: Show loss components
                if args.mode == "binary":
                    if 'selectivity_loss' in train_losses:  # HYBRID LOSS
                        print(f"  🎯 MOVE RECALL LOSS: {train_losses['move_recall_loss']:.4f} (Catch moves)")
                        print(f"  📊 MOVE PRECISION LOSS: {train_losses['move_precision_loss']:.4f} (Be accurate)")
                        print(f"  🎚️  SELECTIVITY LOSS: {train_losses['selectivity_loss']:.4f} (Target 15% moves)")
                        print(f"  ⭐ CONFIDENCE BONUS: {train_losses['confidence_bonus']:.4f} (High-conf rewards)")
                        print(f"  Move Recall: {train_losses.get('move_recall', 0):.4f} (% of moves caught)")
                        print(f"  Move Precision: {train_losses.get('move_precision', 0):.4f} (% predictions correct)")
                        print(f"  Actual Move Ratio: {train_losses.get('actual_move_ratio', 0):.4f} (Target: 0.15)")
                    elif 'move_recall_loss' in train_losses:
                        print(f"  🎯 MOVE RECALL LOSS: {train_losses['move_recall_loss']:.4f} (PRIMARY GOAL)")
                        print(f"  Move Precision Loss: {train_losses['move_precision_loss']:.4f}")
                        print(f"  No-Move Accuracy Loss: {train_losses['no_move_accuracy_loss']:.4f}")
                        print(f"  Move Recall: {train_losses.get('move_recall', 0):.4f} (% of moves caught)")
                        print(f"  Move Precision: {train_losses.get('move_precision', 0):.4f} (% predictions correct)")
                        print(f"  No-Move Accuracy: {train_losses.get('no_move_accuracy', 0):.4f}")
                    elif 'move_reward' in train_losses:
                        print(f"  Move Reward: {train_losses['move_reward']:.4f}")
                        print(f"  False Positive Penalty: {train_losses['false_positive_penalty']:.4f}")
                        print(f"  False Negative Penalty: {train_losses['false_negative_penalty']:.4f}")
                        print(f"  Precision: {train_losses.get('precision', 0):.4f}")
                        print(f"  Recall: {train_losses.get('recall', 0):.4f}")
                        print(f"  F1 Score: {train_losses.get('f1_score', 0):.4f}")
                    elif 'move_focal' in train_losses:
                        print(f"  Train Focal Loss: {train_losses['move_focal']:.4f}")
                        print(f"  Train Label Smooth: {train_losses['move_label_smooth']:.4f}")
                else:
                    if 'direction_focal' in train_losses:
                        print(f"  Train Focal Loss: {train_losses['direction_focal']:.4f}")
                    if 'direction_label_smooth' in train_losses:
                        print(f"  Train Label Smooth: {train_losses['direction_label_smooth']:.4f}")
                    if 'entropy_reg' in train_losses:
                        print(f"  Train Entropy Reg: {train_losses['entropy_reg']:.4f}")
                
                print(f"  LR: {current_lr:.8f}")
            
            # Early stopping based on accuracy
            if args.mode == "binary":
                current_acc = val_metrics['move_accuracy']
                if current_acc > best_direction_acc:
                    best_direction_acc = current_acc
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                    # Save best model
                    model_path = f"models/best_model_split_{split_idx}.pth"
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_losses['total'],
                        'val_metrics': val_metrics,
                        'args': args,
                        'feature_cols': feature_cols,
                        'robust_scaler': feature_scaler,
                        'standard_scaler': standard_scaler
                    }, model_path)
                    print(f"  💾 Saved best model: {model_path} (Acc: {current_acc:.4f})")
                else:
                    patience_counter += 1
            else:
                current_direction_acc = val_metrics['direction_accuracy']
                if current_direction_acc > best_direction_acc:
                    best_direction_acc = current_direction_acc
                    best_val_loss = val_losses['total']
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': val_losses['total'],
                        'val_metrics': val_metrics,
                        'args': args,
                        'feature_cols': feature_cols,
                        'robust_scaler': feature_scaler,
                        'standard_scaler': standard_scaler
                    }, f"models/best_model_split_{split_idx}.pth")
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model and evaluate on test set
        model_path = f"models/best_model_split_{split_idx}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model for split {split_idx} with validation loss: {checkpoint['val_loss']:.4f}")
        else:
            print(f"No checkpoint found for split {split_idx}, using current model state")
        
        test_losses, test_predictions, test_metrics = evaluate_model(model, test_loader, 
                                                                   criterion, device)
        
        print(f"\nTest Results for Split {split_idx + 1}:")
        if args.mode == "binary":
            print(f"  Move Accuracy: {test_metrics['move_accuracy']:.4f}")
            print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
            print(f"  Recall: {test_metrics.get('recall', 0):.4f}")
            print(f"  F1 Score: {test_metrics.get('f1_score', 0):.4f}")
            true_dist = test_metrics.get('true_distribution', {})
            pred_dist = test_metrics.get('predicted_distribution', {})
            print(f"  True Distribution: No Move={true_dist.get('no_move', 0)}, Expected Move={true_dist.get('expected_move', 0)}")
            print(f"  Predicted Distribution: No Move={pred_dist.get('no_move', 0)}, Expected Move={pred_dist.get('expected_move', 0)}")
        else:
            print(f"  Direction Accuracy: {test_metrics['direction_accuracy']:.4f}")
            print(f"  Class Distribution: {test_metrics.get('class_distribution', {})}")
            print(f"  Short Accuracy: {test_metrics.get('class_0_accuracy', 0):.4f}")
            print(f"  Neutral Accuracy: {test_metrics.get('class_1_accuracy', 0):.4f}")
            print(f"  Long Accuracy: {test_metrics.get('class_2_accuracy', 0):.4f}")
        
        all_results.append({
            'split': split_idx,
            'test_losses': test_losses,
            'test_metrics': test_metrics,
            'best_epoch': checkpoint['epoch']
        })
    
    # Summary results
    print(f"\n=== Overall Results ===")
    
    if args.mode == "binary":
        avg_accuracy = np.mean([r['test_metrics']['move_accuracy'] for r in all_results])
        print(f"Average Move Accuracy: {avg_accuracy:.4f}")
        
        # Find best model based on move accuracy
        best_split_idx = max(range(len(all_results)), 
                            key=lambda i: all_results[i]['test_metrics']['move_accuracy'])
        best_accuracy = all_results[best_split_idx]['test_metrics']['move_accuracy']
        
        print(f"\nBest model: Split {best_split_idx} with Move Accuracy {best_accuracy:.4f}")
        
        # Additional binary metrics
        avg_precision = np.mean([r['test_metrics'].get('precision', 0) for r in all_results])
        avg_recall = np.mean([r['test_metrics'].get('recall', 0) for r in all_results])
        avg_f1 = np.mean([r['test_metrics'].get('f1_score', 0) for r in all_results])
        
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
    else:
        avg_accuracy = np.mean([r['test_metrics']['direction_accuracy'] for r in all_results])
        print(f"Average Direction Accuracy: {avg_accuracy:.4f}")
        
        # Find best model based on direction accuracy
        best_split_idx = max(range(len(all_results)), 
                            key=lambda i: all_results[i]['test_metrics']['direction_accuracy'])
        best_accuracy = all_results[best_split_idx]['test_metrics']['direction_accuracy']
        
        print(f"\nBest model: Split {best_split_idx} with Direction Accuracy {best_accuracy:.4f}")
    
    # Copy best model to final location
    import shutil
    if args.mode == "binary":
        final_model_name = f"models/best_binary_{args.symbol}_{args.interval}.pth"
    else:
        final_model_name = f"models/best_directional_{args.symbol}_{args.interval}.pth"
    
    shutil.copy(f"models/best_model_split_{best_split_idx}.pth", final_model_name)
    
    # Save comprehensive results
    if args.mode == "binary":
        results = {
            'args': vars(args),
            'results': all_results,
            'summary': {
                'avg_move_accuracy': avg_accuracy,
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'avg_f1_score': avg_f1,
                'best_split': best_split_idx,
                'best_move_accuracy': best_accuracy,
                'total_parameters': total_params
            }
        }
    else:
        results = {
            'args': vars(args),
            'results': all_results,
            'summary': {
                'avg_direction_accuracy': avg_accuracy,
                'best_split': best_split_idx,
                'best_direction_accuracy': best_accuracy,
                'total_parameters': total_params
            }
        }
    
    if args.mode == "binary":
        results_filename = f"binary_results_{args.symbol}_{args.interval}.json"
        model_filename = f"models/best_binary_{args.symbol}_{args.interval}.pth"
    else:
        results_filename = f"directional_results_{args.symbol}_{args.interval}.json"
        model_filename = f"models/best_directional_{args.symbol}_{args.interval}.pth"
    
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {results_filename}")
    print(f"Best model saved to {model_filename}")

if __name__ == "__main__":
    main()