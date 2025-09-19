import asyncio
import logging
import time
import json
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import RobustScaler
import sys
import os

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directories to path for imports
sys.path.append('../../')
sys.path.append('../../data_prep/')

# Model architectures embedded directly (no external dependencies)
import math

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
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        
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
            'attention_weights': attention_weights[-1] if attention_weights else None
        }
        
        return outputs

class TradingDirectionalModel(nn.Module):
    """Phase 2: Trading-grade directional model for predicting move direction."""
    
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
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x)
        
        # Attention pooling to get sequence representation
        attention_weights = self.attention_pooling(x)  # (batch_size, seq_len, 1)
        pooled = torch.sum(x * attention_weights, dim=1)  # (batch_size, d_model)
        
        # Direction prediction
        direction_logits = self.direction_head(pooled)
        
        return {
            'direction': direction_logits,
            'attention_weights': attention_weights
        }

logger.info("[AI] Model architectures embedded successfully")

class COMClient:
    def __init__(self, base_url: str, api_key: str, secret_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.secret_key = secret_key
        self.session = requests.Session()
        
    def create_hmac_signature(self, timestamp: int, method: str, path: str, body: str) -> str:
        base_string = f"{timestamp}\n{method}\n{path}\n{body}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            base_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def create_order(self, symbol: str, side: str, order_type: str = "MARKET", price: Optional[float] = None) -> Dict:
        method = "POST"
        path = "/api/v1/orders/orders"
        timestamp = int(time.time())
        
        payload = {
            "idempotency_key": f"LTSM_v1_{timestamp}_{symbol}_{side}_{int(time.time() * 1000000) % 1000000}",
            "environment": {"sandbox": True},  # PAPER TRADING
            "source": {
                "strategy_id": "LTSM",
                "instance_id": "instance_001", 
                "owner": "live_trader"
            },
            "order": {
                "instrument": {"class": "crypto_perp", "symbol": "DOGE_USDT"},  # Use underscore format per docs
                "side": side,
                "order_type": order_type,
                "time_in_force": "GTC",
                "flags": {
                    "post_only": order_type == "LIMIT",  # Post-only maker orders per user preference
                    "reduce_only": False,
                    "hidden": False,
                    "iceberg": {},
                    "allow_partial_fills": True
                },
                "routing": {"mode": "AUTO"},
                "leverage": {"enabled": True, "leverage": 25.0},
                "risk": {
                    "sizing": {
                        "mode": "PCT_BALANCE",
                        "value": 10.0,  # 10% of balance
                        "cap": {"notional": 10000.0},
                        "floor": {"notional": 10.0}
                    }
                }
            }
        }
        
        if price and order_type == "LIMIT":
            payload["order"]["price"] = price
        
        body = json.dumps(payload)
        signature = self.create_hmac_signature(timestamp, method, path, body)
        
        headers = {
            "Authorization": f'HMAC key_id="{self.api_key}", signature="{signature}", ts={timestamp}',
            "Content-Type": "application/json"
        }
        
        try:
            url = f"{self.base_url}{path}"
            logger.info(f"[COM] Sending PAPER TRADE order: {order_type} {side} {symbol}")
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[SUCCESS] PAPER ORDER created: {result}")
                return result
            else:
                logger.error(f"[ERROR] PAPER ORDER failed: {response.status_code} - {response.text}")
                return {"error": response.text}
        except Exception as e:
            logger.error(f"[ERROR] COM request failed: {e}")
            return {"error": str(e)}

class ModelInference:
    def __init__(self, binary_model_path: str, directional_model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self.binary_model = self.load_model(binary_model_path)
        self.directional_model = self.load_model(directional_model_path)
        
        # Load scalers and feature info from binary model
        self.binary_scaler = None
        self.directional_scaler = None
        self.feature_cols = None
        
        if self.binary_model:
            try:
                binary_checkpoint = torch.load(binary_model_path, map_location=self.device, weights_only=False)
                self.binary_scaler = binary_checkpoint.get('robust_scaler')
                self.feature_cols = binary_checkpoint.get('feature_cols', [])
                logger.info(f"[AI] Binary model loaded with {len(self.feature_cols)} features")
                logger.info(f"[DEBUG] Binary scaler loaded: {self.binary_scaler is not None}")
                if self.binary_scaler:
                    logger.info(f"[DEBUG] Scaler type: {type(self.binary_scaler)}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load binary scalers: {e}")
        
        # Store for later use by feature engine
        self.binary_scaler_loaded = self.binary_scaler
        self.feature_cols_loaded = self.feature_cols
        
        logger.info(f"[AI] Models loaded on device: {self.device}")
    
    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    model = checkpoint['model']
                elif 'model_state_dict' in checkpoint:
                    # Reconstruct model from saved args
                    args = checkpoint.get('args')
                    if args is None:
                        logger.error(f"[ERROR] No args found in {model_path}")
                        return None
                    
                    # Determine model type and reconstruct
                    if 'binary' in model_path.lower():
                        if BinaryMoveModel is None:
                            logger.error("[ERROR] BinaryMoveModel class not available")
                            return None
                        model = BinaryMoveModel(
                            input_dim=38,  # 37 features + 1 y_actionable
                            d_model=args.d_model,
                            num_heads=args.num_heads,
                            num_layers=args.num_layers,
                            d_ff=args.d_ff,
                            dropout=args.dropout
                        )
                    else:
                        if TradingDirectionalModel is None:
                            logger.error("[ERROR] TradingDirectionalModel class not available")
                            return None
                        model = TradingDirectionalModel(
                            input_dim=238,  # Advanced features count
                            d_model=args.d_model,
                            num_heads=args.num_heads,
                            num_layers=args.num_layers,
                            d_ff=args.d_ff,
                            dropout=args.dropout
                        )
                    
                    # Load state dict
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    logger.error(f"[ERROR] Unknown checkpoint format for {model_path}")
                    return None
            else:
                model = checkpoint
            
            # Move model to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            logger.info(f"[SUCCESS] Loaded model: {model_path}")
            return model
        except Exception as e:
            logger.error(f"[ERROR] Failed to load model {model_path}: {e}")
            return None
    
    def predict_move(self, features: np.ndarray) -> Tuple[bool, float]:
        if self.binary_model is None:
            logger.error("[ERROR] Binary model not loaded - cannot make predictions")
            return False, 0.0
            
        try:
            with torch.no_grad():
                # DEBUG: Check input features
                logger.info(f"[DEBUG] Input features shape: {features.shape}")
                logger.info(f"[DEBUG] Features stats: min={features.min():.6f}, max={features.max():.6f}, mean={features.mean():.6f}")
                logger.info(f"[DEBUG] Features sample: {features[:5]}")
                
                # BACKTESTER METHOD: Use actual sequence of features (60 timesteps)
                # features is already a (60, 38) sequence from get_binary_features
                
                features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # (1, 60, 38)
                logger.info(f"[DEBUG] Tensor shape (CORRECTED): {features_tensor.shape}")
                
                outputs = self.binary_model(features_tensor)
                move_logits = outputs['move']  # Get move logits from output dict
                
                logger.info(f"[DEBUG] Model output logits: {move_logits}")
                logger.info(f"[DEBUG] Logits shape: {move_logits.shape}")
                
                move_prediction = torch.argmax(move_logits, dim=1)[0].item()  # 0 or 1
                move_prob = torch.softmax(move_logits, dim=1)[0, 1].item()
                
                logger.info(f"[DEBUG] Softmax probs: {torch.softmax(move_logits, dim=1)[0]}")
                logger.info(f"[DEBUG] Argmax result: {move_prediction}")
                
            return move_prediction == 1, move_prob
        except Exception as e:
            logger.error(f"[ERROR] Binary model prediction failed: {e}")
            return False, 0.0
    
    def predict_direction(self, features: np.ndarray) -> Tuple[str, float]:
        if self.directional_model is None:
            logger.error("[ERROR] Directional model not loaded - cannot make predictions")
            return "HOLD", 0.5
            
        try:
            with torch.no_grad():
                # Model expects (batch_size, seq_len, input_dim)
                # We have 238 features, treat as 1 timestep with 238 features
                features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 238)
                
                outputs = self.directional_model(features_tensor)
                direction_logits = outputs['direction']  # Get direction logits from output dict
                
                direction_prediction = torch.argmax(direction_logits, dim=1)[0].item()  # 0=DOWN, 1=UP
                direction_probs = torch.softmax(direction_logits, dim=1)[0]
                confidence = direction_probs.max().item()
                
                direction = "BUY" if direction_prediction == 1 else "SELL"
                
            return direction, confidence
        except Exception as e:
            logger.error(f"[ERROR] Directional model prediction failed: {e}")
            return "HOLD", 0.5

class AdvancedFeatureEngine:
    def __init__(self, sequence_length: int = 90):
        self.sequence_length = sequence_length
        self.ohlcv_buffer = pd.DataFrame()
        self.trades_buffer = pd.DataFrame()
        
        # Hardcoded feature columns (no parquet dependency)
        self.feature_columns = [
            "open", "high", "low", "close", "volume", "quote_volume",
            "r1", "r2", "r5", "r10", "range_pct", "body_pct", "atr_pct", "rv",
            "vol_z", "avg_trade_size", "buy_vol", "sell_vol", "tot_vol", 
            "mean_size", "max_size", "p95_size", "n_trades", "signed_vol", 
            "imb_aggr", "dCVD", "CVD", "signed_volatility", "block_trades",
            "impact_proxy", "vw_tick_return", "vol_regime", "drawdown", 
            "minute_sin", "minute_cos", "day_sin", "day_cos", 
            "session_asia", "session_europe", "session_us",
            "price_position", "vol_concentration", "vol_entropy"
        ]
        logger.info(f"[FEATURES] Using {len(self.feature_columns)} hardcoded feature columns")
        
    def update_data(self, ohlcv_file: str, trades_file: str):
        ohlcv_ok = self.update_ohlcv(ohlcv_file)
        trades_ok = self.update_trades(trades_file)
        return ohlcv_ok
        
    def update_ohlcv(self, ohlcv_file: str):
        try:
            df = pd.read_csv(ohlcv_file)
            if len(df) > 0:
                df['datetime'] = pd.to_datetime(df['DateTime'])
                df['open'] = df['Open']
                df['high'] = df['High'] 
                df['low'] = df['Low']
                df['close'] = df['Close']
                df['volume'] = df['Volume']
                df['ts'] = df['Timestamp']
                self.ohlcv_buffer = df.sort_values('datetime').tail(200)  # Keep more for indicators
                return True
        except Exception as e:
            logger.error(f"[ERROR] OHLCV update failed: {e}")
            return False
    
    def update_trades(self, trades_file: str):
        try:
            df = pd.read_csv(trades_file)
            if len(df) > 0:
                df['datetime'] = pd.to_datetime(df['DateTime'])
                self.trades_buffer = df.sort_values('datetime').tail(1000)
                return True
        except Exception as e:
            logger.error(f"[ERROR] Trades update failed: {e}")
            return False
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical indicators"""
        if len(df) < 50:
            return df
            
        try:
            # Price indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['ema_12'] = df['close'].ewm(span=12).mean()
            df['ema_26'] = df['close'].ewm(span=26).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Returns
            df['returns_1'] = df['close'].pct_change(1)
            df['returns_5'] = df['close'].pct_change(5)
            df['volatility'] = df['returns_1'].rolling(20).std()
            
            return df
        except Exception as e:
            logger.error(f"[ERROR] Feature calculation failed: {e}")
            return df
    
    def get_binary_features(self) -> Optional[np.ndarray]:
        """Get features for binary model using EXACT same method as backtester"""
        if len(self.ohlcv_buffer) < 90:  # Need enough data for feature calculation
            return None
            
        try:
            # Calculate features EXACTLY like build_features.py
            df = self.ohlcv_buffer.tail(90).copy()  # Use enough data for rolling calculations
            
            # 1. Core OHLCV Features (EXACT same as build_features.py)
            df["r1"] = np.log(df["close"]).diff()
            df["r2"] = df["r1"].rolling(2).sum()
            df["r5"] = df["r1"].rolling(5).sum()
            df["r10"] = df["r1"].rolling(10).sum()
            
            df["range_pct"] = (df["high"] - df["low"]) / df["close"].shift(1)
            df["body_pct"] = abs(df["close"] - df["open"]) / df["close"].shift(1)
            df["atr_pct"] = (df["high"].combine(df["low"], max) - 
                           df["low"].combine(df["high"], min)) / df["close"].shift(1)
            df["rv"] = df["r1"].pow(2)
            
            # Volume z-score (simplified - no rolling for live)
            df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
            df["avg_trade_size"] = df["volume"] / 100  # Simplified since we don't have trade count
            
            # Trade features (simplified since we don't have full trade data)
            df["buy_vol"] = df["volume"] * 0.5  # Assume 50/50 split
            df["sell_vol"] = df["volume"] * 0.5
            df["tot_vol"] = df["volume"]
            df["mean_size"] = df["volume"] / 100
            df["max_size"] = df["volume"] / 50
            df["p95_size"] = df["volume"] / 60
            df["n_trades"] = 100.0  # Estimated
            df["signed_vol"] = 0.0  # Neutral
            df["imb_aggr"] = 0.0  # Neutral
            df["dCVD"] = 0.0
            df["CVD"] = 0.0
            df["signed_volatility"] = 0.0
            df["block_trades"] = 0.0
            df["impact_proxy"] = abs(df["r1"]) / (df["volume"] + 1e-9)
            df["vw_tick_return"] = 0.0
            df["vol_regime"] = 1.0  # Medium regime
            
            # Drawdown
            rolling_max = df["close"].rolling(20).max()
            df["drawdown"] = (df["close"] - rolling_max) / (rolling_max + 1e-9)
            
            # Calendar features
            df["minute_of_day"] = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["minute_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / (24 * 60))
            df["minute_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / (24 * 60))
            df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
            
            # Session tags
            hour = df["datetime"].dt.hour
            df["session_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
            df["session_europe"] = ((hour >= 8) & (hour < 16)).astype(int)
            df["session_us"] = ((hour >= 16) & (hour < 24)).astype(int)
            
            # Price position
            df["price_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
            df["vol_concentration"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)
            df["vol_entropy"] = -df["vol_concentration"] * np.log(df["vol_concentration"] + 1e-12)
            
            # Use the EXACT feature columns the model was trained on
            if hasattr(self, 'feature_cols') and self.feature_cols:
                model_feature_cols = self.feature_cols
                logger.info(f"[DEBUG] Model expects {len(model_feature_cols)} features from checkpoint")
            else:
                # Fallback to hardcoded feature columns (38 features for binary model)
                model_feature_cols = [
                    "open", "high", "low", "close", "volume", "quote_volume",
                    "r1", "r2", "r5", "r10", "range_pct", "body_pct", "atr_pct", "rv",
                    "vol_z", "avg_trade_size", "buy_vol", "sell_vol", "tot_vol", 
                    "mean_size", "max_size", "p95_size", "n_trades", "signed_vol", 
                    "imb_aggr", "dCVD", "CVD", "signed_volatility", "block_trades",
                    "impact_proxy", "vw_tick_return", "vol_regime", "drawdown", 
                    "minute_sin", "minute_cos", "day_sin", "day_cos"
                ]  # 37 features (38 with ts, but we don't include ts in feature matrix)
                logger.info(f"[DEBUG] Using hardcoded {len(model_feature_cols)} features (fallback)")
            
            # Get last 60 rows for sequence (like backtester)
            sequence_data = df.tail(60).copy()
            
            # Create feature matrix with ONLY the features the model expects
            feature_matrix = np.zeros((60, len(model_feature_cols)))
            
            for i, col in enumerate(model_feature_cols):
                if col in sequence_data.columns:
                    feature_matrix[:, i] = sequence_data[col].fillna(0.0).values
                else:
                    logger.warning(f"[WARNING] Missing feature column: {col}")
                    feature_matrix[:, i] = 0.0  # Missing features as zeros
            
            # Apply the saved scaler (transform the entire sequence)
            if hasattr(self, 'binary_scaler') and self.binary_scaler is not None:
                # Create DataFrame for scaler
                temp_df = pd.DataFrame(feature_matrix, columns=model_feature_cols)
                
                # Apply scaler directly (same as prepare_features does)
                features_scaled = self.binary_scaler.transform(temp_df[model_feature_cols])
                
                logger.info(f"[DEBUG] Used PROPER scaling on 60-timestep sequence")
                logger.info(f"[DEBUG] Sequence shape: {features_scaled.shape}")
                
                return features_scaled  # Return the full sequence (60, 38)
            else:
                logger.warning("[WARNING] No binary scaler available, using raw features")
                return feature_matrix
            
        except Exception as e:
            logger.error(f"[ERROR] Binary feature extraction failed: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def get_directional_features(self) -> Optional[np.ndarray]:
        """Get advanced features for directional model (238 features)"""
        if len(self.ohlcv_buffer) < 37:
            return None
            
        try:
            # Use the first 37 close prices (same as binary) and pad to 238
            recent_closes = self.ohlcv_buffer['close'].tail(37).values
            
            # Simple normalization
            if np.std(recent_closes) > 0:
                features = (recent_closes - np.mean(recent_closes)) / np.std(recent_closes)
            else:
                features = recent_closes
            
            # Pad to exactly 238 features (as expected by directional model)
            if len(features) < 238:
                padding = np.zeros(238 - len(features))
                features = np.concatenate([features, padding])
            else:
                features = features[:238]
            
            logger.info(f"[FEATURES] Directional features prepared: {len(features)} features")
            return features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"[ERROR] Directional feature extraction failed: {e}")
            return None

class PositionManager:
    def __init__(self, com_client: COMClient):
        self.com_client = com_client
        self.positions = {}
        
    def open_position(self, symbol: str, side: str, current_price: float) -> Optional[str]:
        try:
            # Create market entry order (PAPER TRADE)
            logger.info(f"[COM] Sending PAPER TRADE market order: {side} {symbol}")
            order_response = self.com_client.create_order(symbol, side, "MARKET")
            
            if "error" in order_response:
                logger.error(f"[ERROR] Failed to open PAPER position: {order_response['error']}")
                return None
            
            position_ref = order_response.get("position_ref", f"demo_pos_{int(time.time())}")
            
            # Calculate TP price (0.35% away)
            tp_multiplier = 1.0035 if side == "BUY" else 0.9965
            tp_price = current_price * tp_multiplier
            
            # Create TP order (post-only limit order)
            tp_side = "SELL" if side == "BUY" else "BUY"
            logger.info(f"[COM] Sending PAPER TRADE TP order: {tp_side} {symbol} @ {tp_price:.6f}")
            tp_response = self.com_client.create_order(symbol, tp_side, "LIMIT", tp_price)
            
            # Store position info
            self.positions[position_ref] = {
                "symbol": symbol,
                "side": side,
                "entry_price": current_price,
                "tp_price": tp_price,
                "entry_time": datetime.now(),
                "status": "OPEN"
            }
            
            logger.info(f"[POSITION] PAPER TRADE OPENED: {side} {symbol} @ {current_price:.6f}, TP: {tp_price:.6f}")
            return position_ref
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to open position: {e}")
            return None
    
    def check_stops(self, max_hold_minutes: int = 30, current_price: float = None):
        """Check both time stops and price stops (-2% SL)"""
        current_time = datetime.now()
        for position_id, position in list(self.positions.items()):
            if position["status"] != "OPEN":
                continue
                
            # Time stop check
            hold_time = current_time - position["entry_time"]
            if hold_time.total_seconds() > (max_hold_minutes * 60):
                logger.info(f"[TIME_STOP] Closing position {position_id} after {max_hold_minutes} minutes")
                position["status"] = "CLOSED"
                position["close_reason"] = "TIME_STOP"
                continue
            
            # Price stop loss check (-2%)
            if current_price is not None:
                entry_price = position["entry_price"]
                side = position["side"]
                
                # Calculate -2% stop loss level
                if side == "BUY":
                    stop_loss_price = entry_price * 0.98  # -2% for long positions
                    if current_price <= stop_loss_price:
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                        logger.info(f"[STOP_LOSS] Closing BUY position {position_id}: ${current_price:.6f} <= ${stop_loss_price:.6f} (P&L: {pnl_pct:.2f}%)")
                        position["status"] = "CLOSED"
                        position["close_reason"] = "STOP_LOSS"
                        position["close_price"] = current_price
                        position["pnl_pct"] = pnl_pct
                else:  # SELL
                    stop_loss_price = entry_price * 1.02  # +2% for short positions
                    if current_price >= stop_loss_price:
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                        logger.info(f"[STOP_LOSS] Closing SELL position {position_id}: ${current_price:.6f} >= ${stop_loss_price:.6f} (P&L: {pnl_pct:.2f}%)")
                        position["status"] = "CLOSED"
                        position["close_reason"] = "STOP_LOSS"
                        position["close_price"] = current_price
                        position["pnl_pct"] = pnl_pct
    
    def get_open_positions_count(self) -> int:
        return len([p for p in self.positions.values() if p["status"] == "OPEN"])

class LiveDOGETrader:
    def __init__(self, config: Dict):
        self.com_client = COMClient(
            base_url=config["com_base_url"],
            api_key=config["api_key"],
            secret_key=config["secret_key"]
        )
        
        self.model_inference = ModelInference(
            binary_model_path=config["binary_model_path"],
            directional_model_path=config["directional_model_path"]
        )
        
        self.feature_engine = AdvancedFeatureEngine(sequence_length=config["sequence_length"])
        self.position_manager = PositionManager(self.com_client)
        
        # Pass the binary scaler and feature columns to feature engine
        self.feature_engine.binary_scaler = getattr(self.model_inference, 'binary_scaler_loaded', None)
        self.feature_engine.feature_cols = getattr(self.model_inference, 'feature_cols_loaded', [])
        
        self.symbol = config["symbol"]
        self.max_positions = config.get("max_positions", 1)
        self.max_hold_minutes = config.get("max_hold_minutes", 30)
        self.ohlcv_file = config["ohlcv_file"]
        self.trades_file = config["trades_file"]
        
        self.running = False
        self.last_candle_time = None
        
        logger.info("[START] LiveDOGETrader initialized with PAPER TRADING")
    
    async def run(self):
        self.running = True
        logger.info("[SIGNAL] Starting live PAPER trading...")
        
        while self.running:
            try:
                # Update data from CSV files
                data_updated = self.feature_engine.update_data(self.ohlcv_file, self.trades_file)
                
                if not data_updated:
                    logger.warning("[WARNING] Failed to update data")
                    await asyncio.sleep(10)
                    continue
                
                # Check for new candle
                if len(self.feature_engine.ohlcv_buffer) > 0:
                    latest_candle_time = self.feature_engine.ohlcv_buffer['datetime'].iloc[-1]
                    
                    # Process new candle
                    if self.last_candle_time is None or latest_candle_time > self.last_candle_time:
                        self.last_candle_time = latest_candle_time
                        await self.process_new_candle()
                
                # Check time stops and price stops (-2% SL)
                current_price = self.feature_engine.ohlcv_buffer['close'].iloc[-1] if len(self.feature_engine.ohlcv_buffer) > 0 else None
                self.position_manager.check_stops(self.max_hold_minutes, current_price)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                logger.info("[STOP] Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"[ERROR] Error in main loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("[FINISH] PAPER trading stopped")
    
    async def process_new_candle(self):
        """Process each new candle with full feature window"""
        try:
            logger.info(f"[CANDLE] New candle detected at {self.last_candle_time}")
            
            # Get current price
            current_price = self.feature_engine.ohlcv_buffer['close'].iloc[-1]
            
            # Phase 1: Binary model with basic features (37+1=38)
            binary_features = self.feature_engine.get_binary_features()
            if binary_features is None:
                logger.warning("[WARNING] No binary features available")
                return
            
            will_move, move_confidence = self.model_inference.predict_move(binary_features)
            
            logger.info(f"[AI] Current Price: ${current_price:.6f}")
            logger.info(f"[AI] Move prediction: {will_move} (confidence: {move_confidence:.3f})")
            
            if not will_move:
                logger.info("[DATA] No move predicted for this candle")
                return
            
            # Phase 2: Directional model with advanced features (232)
            directional_features = self.feature_engine.get_directional_features()
            if directional_features is None:
                logger.warning("[WARNING] No directional features available")
                return
            
            direction, direction_confidence = self.model_inference.predict_direction(directional_features)
            
            logger.info(f"[AI] Direction prediction: {direction} (confidence: {direction_confidence:.3f})")
            
            if direction == "HOLD":
                logger.info("[DATA] No clear direction")
                return
            
            # Check position limits
            open_positions = self.position_manager.get_open_positions_count()
            if open_positions >= self.max_positions:
                logger.info(f"[DATA] Max positions reached ({open_positions}/{self.max_positions})")
                return
            
            # Calculate TP and SL
            tp_multiplier = 1.0035 if direction == "BUY" else 0.9965
            tp_price = current_price * tp_multiplier
            sl_price = current_price * 0.98 if direction == "BUY" else current_price * 1.02
            
            # Display signal
            logger.info("[SIGNAL] === TRADING SIGNAL ===")
            logger.info(f"[SIGNAL] Action: {direction} DOGE/USDT")
            logger.info(f"[SIGNAL] Entry: ${current_price:.6f}")
            logger.info(f"[SIGNAL] TP Target: ${tp_price:.6f} (0.35%)")
            logger.info(f"[SIGNAL] SL Level: ${sl_price:.6f} (-2.00%)")
            logger.info(f"[SIGNAL] Move Confidence: {move_confidence:.1%}")
            logger.info(f"[SIGNAL] Direction Confidence: {direction_confidence:.1%}")
            logger.info(f"[SIGNAL] Trade Type: PAPER TRADE")
            logger.info("[SIGNAL] ========================")
            
            # Execute PAPER trade
            position_id = self.position_manager.open_position(self.symbol, direction, current_price)
            
            if position_id:
                logger.info(f"[SUCCESS] PAPER position opened: {position_id}")
            else:
                logger.error("[ERROR] Failed to open PAPER position")
                
        except Exception as e:
            logger.error(f"[ERROR] Error processing new candle: {e}")

def load_config() -> Dict:
    return {
        "com_base_url": "http://localhost:8000",
        "api_key": "UkFWRl8yMDI1MDkxOF8yMDU4NDVAibQoNKUlDR_SmPzIYHxui1E4kNPrpCmQADqKdu7pvw",
        "secret_key": "1u28c-DX3vlys3u0-iNWFvRSnoy420jKDolC2bzH84Gc7HL2C1m5N4Gur7YEn4SdtN-ZN7n3E2YmQaDtirCSNw",
        "salt": "_L1X-cWBuqhu08NeBbv97gfTo8E_N9ON1TZhHIGiPN4",
        "strategy_name": "LTSM_v1",
        "strategy_id": "LTSM",
        "instance_id": "instance_001",
        "owner": "live_trader",
        "binary_model_path": "models/binary.pth",
        "directional_model_path": "models/directonal.pth",
        "ohlcv_file": "datadoge.csv",
        "trades_file": "aggtradesdoge.csv",
        "symbol": "DOGE_USDT",
        "sequence_length": 90,
        "max_positions": 1,
        "max_hold_minutes": 30,
    }

async def main():
    logger.info("[START] Starting Live DOGE PAPER Trader")
    
    config = load_config()
    trader = LiveDOGETrader(config)
    
    try:
        await trader.run()
    except KeyboardInterrupt:
        logger.info("[STOP] Shutting down...")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())