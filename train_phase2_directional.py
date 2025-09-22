#!/usr/bin/env python3
"""
Phase 2: Directional Model Training Script

This script trains a directional model to predict move direction (SHORT vs LONG)
using the high-quality directional labels created by the binary model in Phase 1.

Input: binary_labels_DOGEUSDT_1m.parquet (from build_directional_labels.py)
Output: Trained directional model for Phase 2
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(s: int = 42):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.cuda.manual_seed_all(s)

set_seed(42)
torch.set_float32_matmul_precision('high')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlock(nn.Module):
    """Single transformer block with multi-head attention"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention
        attn_out, attention_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attention_weights

class TradingDirectionalModel(nn.Module):
    """
    Phase 2: Trading-grade directional model for predicting move direction.
    
    Architecture optimized for reliability and profitability:
    - Transformer-based sequence processing
    - Attention pooling for sequence summarization  
    - 4-layer MLP for direction prediction
    - Binary output: 0=SHORT, 1=LONG
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, d_ff: int = 512, dropout: float = 0.2):
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
        
        # Attention pooling for sequence summarization
        self.attention_pooling = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Direction prediction head (4-layer MLP)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Binary: 0=SHORT, 1=LONG
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.shape
        
        # Input projection and positional encoding
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        attention_weights = []
        for layer in self.transformer_layers:
            x, attn_weights = layer(x)
            attention_weights.append(attn_weights)
        
        # Attention pooling to get single representation
        pool_weights = self.attention_pooling(x)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * pool_weights, dim=1)  # (batch_size, d_model)
        
        # Direction prediction
        direction_logits = self.direction_head(pooled)
        
        outputs = {
            'direction': direction_logits,
            'attention_weights': attention_weights[-1],  # Last layer attention
            'pool_weights': pool_weights.squeeze(-1)  # Attention pooling weights
        }
        
        return outputs

class DirectionalDataset(Dataset):
    """Dataset for directional model training - only uses validated directional moves"""
    
    def __init__(self, features: np.ndarray, targets: Dict[str, np.ndarray], 
                 sequence_length: int = 60):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Ensure we have enough data for sequences
        self.valid_length = len(self.features) - self.sequence_length + 1
        if self.valid_length <= 0:
            raise ValueError(f"Not enough data for sequence_length={sequence_length}")
        
    def __len__(self):
        return self.valid_length
    
    def __getitem__(self, idx):
        # Input sequence
        X = self.features[idx:idx + self.sequence_length]
        
        # Target index for current predictions
        target_idx = idx + self.sequence_length - 1
        
        # Direction target: -1=SHORT -> 0, +1=LONG -> 1
        direction = int(self.targets['y_direction'][target_idx])
        direction_binary = 0 if direction == -1 else 1
        
        y = {
            'direction': torch.LongTensor([direction_binary]),
        }
        
        return torch.FloatTensor(X), y

class ForceComplexityLoss(nn.Module):
    """
    AGGRESSIVE loss that FORCES the model to use complex patterns.
    
    Strategy: Make it IMPOSSIBLE for the model to succeed with simple shortcuts.
    Forces the model to develop sophisticated directional understanding.
    """
    
    def __init__(self, ce_weight: float = 1.0,
                 min_both_classes_weight: float = 10.0,
                 representation_diversity_weight: float = 2.0,
                 temporal_consistency_weight: float = 1.0,
                 confidence_calibration_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.min_both_classes_weight = min_both_classes_weight
        self.representation_diversity_weight = representation_diversity_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        self.confidence_calibration_weight = confidence_calibration_weight
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(self, outputs: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        direction_logits = outputs['direction']
        direction_targets = targets['direction'].squeeze()
        
        if direction_targets.dim() == 0:
            direction_targets = direction_targets.unsqueeze(0)
        
        batch_size = direction_logits.shape[0]
        direction_probs = torch.softmax(direction_logits, dim=1)
        direction_preds = torch.argmax(direction_logits, dim=1)
        
        # 1. BASE CROSS-ENTROPY LOSS
        ce_loss = self.ce_loss(direction_logits, direction_targets)
        
        # 2. FORCE BOTH CLASSES: Massive penalty if either class has <20% of predictions
        pred_counts = torch.bincount(direction_preds, minlength=2).float()
        short_ratio = pred_counts[0] / batch_size
        long_ratio = pred_counts[1] / batch_size
        
        # NUCLEAR: If either class is predicted <25% of the time, MASSIVE penalty
        min_class_ratio = torch.min(short_ratio, long_ratio)
        min_both_classes_loss = self.min_both_classes_weight * torch.relu(0.25 - min_class_ratio)
        
        # EXTRA NUCLEAR: If either class is predicted <5%, add EXTREME penalty
        if min_class_ratio < 0.05:
            min_both_classes_loss += 100.0  # Catastrophic penalty for near-complete collapse
        
        # 3. REPRESENTATION DIVERSITY: Force SHORT and LONG to use different internal patterns
        representation_diversity_loss = torch.tensor(0.0, device=direction_logits.device)
        
        short_mask = (direction_targets == 0)
        long_mask = (direction_targets == 1)
        
        if short_mask.sum() > 0 and long_mask.sum() > 0:
            # Get the raw logits (before softmax) for each class
            short_logits = direction_logits[short_mask]  # All SHORT samples
            long_logits = direction_logits[long_mask]    # All LONG samples
            
            # Calculate mean logit patterns for each class
            short_mean_logits = short_logits.mean(dim=0)  # [2] - average SHORT logit pattern
            long_mean_logits = long_logits.mean(dim=0)    # [2] - average LONG logit pattern
            
            # Force the logit patterns to be DIFFERENT
            logit_similarity = torch.cosine_similarity(short_mean_logits.unsqueeze(0), 
                                                     long_mean_logits.unsqueeze(0))
            
            # Penalty for similar internal representations
            representation_diversity_loss = self.representation_diversity_weight * torch.relu(logit_similarity + 0.2)
        
        # 4. TEMPORAL CONSISTENCY: Penalize if predictions change too randomly over time
        # (This would require sequence of predictions, skip for now)
        temporal_consistency_loss = torch.tensor(0.0, device=direction_logits.device)
        
        # 5. CONFIDENCE CALIBRATION: Force model to be confident when correct, uncertain when wrong
        correct_mask = (direction_preds == direction_targets)
        wrong_mask = ~correct_mask
        
        confidence_calibration_loss = torch.tensor(0.0, device=direction_logits.device)
        
        if correct_mask.sum() > 0:
            # For correct predictions, encourage high confidence
            correct_confidences = direction_probs[correct_mask].max(dim=1)[0]
            correct_confidence_loss = -correct_confidences.mean()  # Negative = reward high confidence
        else:
            correct_confidence_loss = torch.tensor(0.0, device=direction_logits.device)
            
        if wrong_mask.sum() > 0:
            # For wrong predictions, encourage low confidence (uncertainty)
            wrong_confidences = direction_probs[wrong_mask].max(dim=1)[0]
            wrong_confidence_loss = wrong_confidences.mean()  # Positive = penalty for high confidence on wrong
        else:
            wrong_confidence_loss = torch.tensor(0.0, device=direction_logits.device)
            
        confidence_calibration_loss = (correct_confidence_loss + wrong_confidence_loss) * self.confidence_calibration_weight
        
        # TOTAL LOSS: Combine all components
        total_loss = (ce_loss * self.ce_weight + 
                     min_both_classes_loss +
                     representation_diversity_loss + 
                     temporal_consistency_loss +
                     confidence_calibration_loss)
        
        # Calculate metrics for monitoring
        with torch.no_grad():
            correct_predictions = (direction_preds == direction_targets).float().mean()
            avg_confidence = direction_probs.max(dim=1)[0].mean()
            
            short_accuracy = ((direction_targets == 0) & (direction_preds == 0)).float().mean()
            long_accuracy = ((direction_targets == 1) & (direction_preds == 1)).float().mean()
            
            # Class balance metrics
            balance_score = 1.0 - torch.abs(short_ratio - 0.5) * 2  # 1.0 = perfect balance, 0.0 = complete imbalance
        
        return {
            'total': total_loss,
            'ce_loss': ce_loss,
            'min_both_classes_loss': min_both_classes_loss,
            'representation_diversity_loss': representation_diversity_loss,
            'temporal_consistency_loss': temporal_consistency_loss,
            'confidence_calibration_loss': confidence_calibration_loss,
            'correct_predictions': correct_predictions,
            'avg_confidence': avg_confidence,
            'short_ratio': short_ratio,
            'long_ratio': long_ratio,
            'short_accuracy': short_accuracy,
            'long_accuracy': long_accuracy,
            'balance_score': balance_score,
            'min_class_ratio': min_class_ratio
        }

def prepare_features(df: pd.DataFrame, feature_cols: List[str], scaler=None, fit_scaler=False):
    """Prepare and scale features"""
    df_reset = df.reset_index(drop=True)
    feature_data = df_reset[feature_cols].copy()
    
    # Handle missing values
    feature_data = feature_data.ffill().fillna(0)
    
    # Volume features scaling
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'signed_vol', 'dCVD', 'CVD']
    
    for col in feature_data.columns:
        if col in volume_features:
            feature_data[col] = feature_data[col] / 1e6
        elif feature_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            p1, p99 = np.percentile(feature_data[col], [1, 99])
            feature_data[col] = feature_data[col].clip(p1, p99)
    
    # Apply scaling
    if scaler is None:
        scaler = RobustScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)
    else:
        if fit_scaler:
            feature_data_scaled = scaler.fit_transform(feature_data)
        else:
            feature_data_scaled = scaler.transform(feature_data)
    
    return feature_data_scaled.astype(np.float32), scaler

def create_directional_targets(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Create directional targets from the directional labels"""
    return {
        'y_direction': df['direction'].values  # -1=SHORT, 1=LONG
    }

def walk_forward_split(df: pd.DataFrame, args, train_months: int, val_months: int, 
                      test_months: int, stride_days: int, max_splits: int):
    """Create walk-forward validation splits"""
    df['date'] = pd.to_datetime(df['ts']).dt.date
    unique_dates = sorted(df['date'].unique())
    
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"Total days: {len(unique_dates)}")
    
    # Calculate required days per split
    days_per_month = 30.44
    train_days = int(train_months * days_per_month)
    val_days = int(val_months * days_per_month)
    test_days = int(test_months * days_per_month)
    total_required = train_days + val_days + test_days
    
    print(f"Required days per split: {total_required}")
    print(f"Train days: {train_days}, Val days: {val_days}, Test days: {test_days}")
    
    if len(unique_dates) < total_required:
        print(f"ERROR: Not enough data! Have {len(unique_dates)} days, need {total_required}")
        return []
    
    splits = []
    
    # Start from the most recent data (end of dataset)
    for split_idx in range(max_splits):
        # Calculate date indices (working backwards from the end)
        end_date_idx = len(unique_dates)
        test_start = end_date_idx - test_days
        val_start = test_start - val_days
        train_start = val_start - train_days
        
        if train_start < 0:
            print(f"Not enough data for split {split_idx}")
            break
        
        # Get date ranges
        train_dates = unique_dates[train_start:val_start]
        val_dates = unique_dates[val_start:test_start]
        test_dates = unique_dates[test_start:end_date_idx]
        
        # Filter data by dates
        train_data = df[df['date'].isin(train_dates)].copy()
        val_data = df[df['date'].isin(val_dates)].copy()
        test_data = df[df['date'].isin(test_dates)].copy()
        
        # Only keep rows with valid directional labels
        train_data = train_data[train_data['valid_direction'] == True].copy()
        val_data = val_data[val_data['valid_direction'] == True].copy()
        test_data = test_data[test_data['valid_direction'] == True].copy()
        
        print(f"Split {split_idx + 1}: Train {train_dates[0]} to {train_dates[-1]}, "
              f"Val {val_dates[0]} to {val_dates[-1]}, Test {test_dates[0]} to {test_dates[-1]}")
        print(f"  Valid samples: Train={len(train_data):,}, Val={len(val_data):,}, Test={len(test_data):,}")
        
        if len(train_data) < 1000 or len(val_data) < 100 or len(test_data) < 100:
            print(f"  Skipping split {split_idx}: Not enough valid directional samples")
            continue
        
        splits.append({
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates
        })
        
        # For now, just use one split
        break
    
    return splits

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model and return metrics"""
    model.eval()
    total_losses = {}
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_features, batch_targets in tqdm(dataloader, desc="Evaluating"):
            batch_features = batch_features.to(device)
            batch_targets = {k: v.to(device) for k, v in batch_targets.items()}
            
            outputs = model(batch_features)
            # Simple cross-entropy loss
            loss = criterion(outputs['direction'], batch_targets['direction'].squeeze())
            losses = {'total': loss}
            
            # Accumulate losses
            for key, loss in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += loss.item()
            
            # Store predictions and targets
            predictions = torch.argmax(outputs['direction'], dim=1).cpu().numpy()
            targets = batch_targets['direction'].squeeze().cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='binary', zero_division=0)
    recall = recall_score(all_targets, all_predictions, average='binary', zero_division=0)
    f1 = f1_score(all_targets, all_predictions, average='binary', zero_division=0)
    
    # Class distributions
    true_counts = np.bincount(all_targets, minlength=2)
    pred_counts = np.bincount(all_predictions, minlength=2)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_distribution': {
            'short': int(true_counts[0]),
            'long': int(true_counts[1])
        },
        'predicted_distribution': {
            'short': int(pred_counts[0]),
            'long': int(pred_counts[1])
        }
    }
    
    # Average losses
    avg_losses = {k: v / len(dataloader) for k, v in total_losses.items()}
    
    return avg_losses, all_predictions, metrics

def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 Directional Model")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--sequence_length", type=int, default=60)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_months", type=int, default=6)
    parser.add_argument("--val_months", type=int, default=1)
    parser.add_argument("--test_months", type=int, default=1)
    parser.add_argument("--stride_days", type=int, default=30)
    parser.add_argument("--max_splits", type=int, default=1)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    
    args = parser.parse_args()
    
    print(f"🎯 Phase 2: Training Directional Model for {args.symbol} {args.interval}")
    print(f"   Architecture: {args.d_model}d, {args.num_heads}h, {args.num_layers}L")
    print(f"   Training: {args.epochs} epochs, {args.batch_size} batch size")
    
    # Load directional labels created by Phase 1
    print(f"Loading Phase 2 directional data...")
    try:
        # Try to load advanced features first, fall back to basic features
        try:
            features_df = pq.read_table(f"features/advanced_features_{args.symbol}_{args.interval}.parquet").to_pandas()
            print(f"✅ Using ADVANCED features: {len(features_df.columns)} columns")
        except FileNotFoundError:
            features_df = pq.read_table(f"features/features_{args.symbol}_{args.interval}.parquet").to_pandas()
            print(f"⚠️  Using basic features: {len(features_df.columns)} columns")
            
        labels_df = pq.read_table(f"features/binary_labels_{args.symbol}_{args.interval}.parquet").to_pandas()
    except FileNotFoundError as e:
        print(f"ERROR: Required files not found: {e}")
        print("Please run build_directional_labels.py first to generate the directional labels.")
        return
    
    # Merge features and directional labels
    df = features_df.merge(labels_df, on='ts', how='inner')
    print(f"Loaded {len(df):,} samples")
    
    # Filter to only valid directional moves (our high-quality Phase 2 data)
    valid_moves = df[df['valid_direction'] == True]
    print(f"Valid directional moves: {len(valid_moves):,} ({len(valid_moves)/len(df)*100:.1f}%)")
    
    if len(valid_moves) < 1000:
        print("ERROR: Not enough valid directional moves for training!")
        return
    
    # Check directional balance
    direction_counts = valid_moves['direction'].value_counts()
    print(f"Direction distribution: SHORT={direction_counts.get(-1, 0):,}, LONG={direction_counts.get(1, 0):,}")
    
    # Get feature columns (exclude directional label columns)
    exclude_cols = ['ts', 'move_detected', 'direction', 'valid_direction', 'entry_price', 'high', 'low', 'close']
    duplicate_features = ['high', 'low', 'close', 'tot_vol', 'atr_pct', 'dCVD']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and col not in duplicate_features]
    
    print(f"Using {len(feature_cols)} features")
    
    # Create walk-forward splits using only valid directional data
    splits = walk_forward_split(df, args, args.train_months, args.val_months, 
                               args.test_months, args.stride_days, args.max_splits)
    
    if len(splits) == 0:
        print("ERROR: No valid splits created!")
        return
    
    print(f"Created {len(splits)} walk-forward splits")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    all_results = []
    
    for split_idx, split in enumerate(splits):
        print(f"\\n=== Phase 2 Split {split_idx + 1}/{len(splits)} ===")
        
        # Prepare features
        train_features, feature_scaler = prepare_features(split['train'], feature_cols, scaler=None, fit_scaler=True)
        val_features, _ = prepare_features(split['val'], feature_cols, scaler=feature_scaler, fit_scaler=False)
        test_features, _ = prepare_features(split['test'], feature_cols, scaler=feature_scaler, fit_scaler=False)
        
        # Apply standard scaling
        standard_scaler = StandardScaler()
        train_features = standard_scaler.fit_transform(train_features)
        val_features = standard_scaler.transform(val_features)
        test_features = standard_scaler.transform(test_features)
        
        # Create directional targets
        train_targets = create_directional_targets(split['train'])
        val_targets = create_directional_targets(split['val'])
        test_targets = create_directional_targets(split['test'])
        
        # Create datasets
        train_dataset = DirectionalDataset(train_features, train_targets, args.sequence_length)
        val_dataset = DirectionalDataset(val_features, val_targets, args.sequence_length)
        test_dataset = DirectionalDataset(test_features, test_targets, args.sequence_length)
        
        print(f"Dataset samples: Train={len(train_dataset):,}, Val={len(val_dataset):,}, Test={len(test_dataset):,}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=True)
        
        # Create model
        model = TradingDirectionalModel(
            input_dim=len(feature_cols),
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            dropout=args.dropout
        ).to(device)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        model.apply(init_weights)
        print("Model weights initialized with Xavier uniform")
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # Warmup scheduler
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Mixed precision scaler
        scaler = GradScaler() if (device.type == 'cuda' and args.mixed_precision) else None
        
        # Simple, proven loss function (what worked before)
        print("🎯 Using SIMPLE CROSS-ENTROPY: Proven stable approach with advanced features!")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"\\n🚀 Starting Phase 2 directional training...")
        
        best_accuracy = 0.0
        patience = 10
        patience_counter = 0
        
        for epoch in range(args.epochs):
            # Training
            model.train()
            total_loss = 0
            
            for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                batch_features = batch_features.to(device)
                batch_targets = {k: v.to(device) for k, v in batch_targets.items()}
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        outputs = model(batch_features)
                        loss = criterion(outputs['direction'], batch_targets['direction'].squeeze())
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_features)
                    loss = criterion(outputs['direction'], batch_targets['direction'].squeeze())
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                
                total_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            val_losses, val_predictions, val_metrics = evaluate_model(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{args.epochs}:")
            print(f"  Train Loss: {total_loss/len(train_loader):.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.1f}%)")
            print(f"  Val Precision: {val_metrics['precision']:.4f} ({val_metrics['precision']*100:.1f}%)")
            print(f"  Val Recall: {val_metrics['recall']:.4f} ({val_metrics['recall']*100:.1f}%)")
            print(f"  Val F1: {val_metrics['f1_score']:.4f} ({val_metrics['f1_score']*100:.1f}%)")
            
            true_dist = val_metrics['true_distribution']
            pred_dist = val_metrics['predicted_distribution']
            print(f"  True: SHORT={true_dist['short']}, LONG={true_dist['long']}")
            print(f"  Pred: SHORT={pred_dist['short']}, LONG={pred_dist['long']}")
            
            # Simple cross-entropy - let the advanced features do the work
            print(f"  📊 Using 239 advanced features with simple, stable cross-entropy loss")
            
            # Early stopping
            current_accuracy = val_metrics['accuracy']
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                patience_counter = 0
                
                # Save best model
                model_path = f"models/best_phase2_split_{split_idx}.pth"
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
                print(f"  💾 Saved best Phase 2 model: {model_path} (Acc: {current_accuracy:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and test
        model_path = f"models/best_phase2_split_{split_idx}.pth"
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best Phase 2 model from epoch {checkpoint['epoch']}")
        
        test_losses, test_predictions, test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        print(f"\\nPhase 2 Test Results for Split {split_idx + 1}:")
        print(f"  Directional Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.1f}%)")
        print(f"  Recall: {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.1f}%)")
        print(f"  F1 Score: {test_metrics['f1_score']:.4f} ({test_metrics['f1_score']*100:.1f}%)")
        
        true_dist = test_metrics['true_distribution']
        pred_dist = test_metrics['predicted_distribution']
        print(f"  True Distribution: SHORT={true_dist['short']}, LONG={true_dist['long']}")
        print(f"  Predicted Distribution: SHORT={pred_dist['short']}, LONG={pred_dist['long']}")
        
        all_results.append({
            'split': split_idx,
            'test_losses': test_losses,
            'test_metrics': test_metrics,
            'best_epoch': checkpoint.get('epoch', epoch) if os.path.exists(model_path) else epoch
        })
    
    # Summary
    print(f"\\n=== Phase 2 Overall Results ===")
    avg_accuracy = np.mean([r['test_metrics']['accuracy'] for r in all_results])
    avg_precision = np.mean([r['test_metrics']['precision'] for r in all_results])
    avg_recall = np.mean([r['test_metrics']['recall'] for r in all_results])
    avg_f1 = np.mean([r['test_metrics']['f1_score'] for r in all_results])
    
    print(f"Average Directional Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.1f}%)")
    print(f"Average Precision: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
    print(f"Average Recall: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f"Average F1 Score: {avg_f1:.4f} ({avg_f1*100:.1f}%)")
    
    # Find best model
    best_split_idx = max(range(len(all_results)), 
                        key=lambda i: all_results[i]['test_metrics']['accuracy'])
    best_accuracy = all_results[best_split_idx]['test_metrics']['accuracy']
    
    print(f"\\nBest Phase 2 model: Split {best_split_idx} with Accuracy {best_accuracy:.4f}")
    
    # Copy best model to final location
    import shutil
    shutil.copy(f"models/best_phase2_split_{best_split_idx}.pth", 
                f"models/best_phase2_directional_{args.symbol}_{args.interval}.pth")
    
    # Save results
    results = {
        'args': vars(args),
        'results': all_results,
        'summary': {
            'avg_directional_accuracy': avg_accuracy,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1_score': avg_f1,
            'best_split': best_split_idx,
            'best_directional_accuracy': best_accuracy,
            'total_parameters': total_params
        }
    }
    
    with open(f"phase2_directional_results_{args.symbol}_{args.interval}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n✅ Phase 2 Complete!")
    print(f"Results saved to phase2_directional_results_{args.symbol}_{args.interval}.json")
    print(f"Best model saved to models/best_phase2_directional_{args.symbol}_{args.interval}.pth")

if __name__ == "__main__":
    main()
