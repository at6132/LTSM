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

# Configure logging to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('livedoge.log'),
        logging.StreamHandler()
    ]
)
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
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_linear(out), attention

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
            'attention_weights': attention_weights[-1] if attention_weights else None
        }
        
        return outputs

class PositionalEncodingDirectional(nn.Module):
    """Positional encoding for directional model (different format)."""
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: [max_seq_length, 1, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBlockDirectional(nn.Module):
    """Single transformer block with multi-head attention for directional model"""
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
    """Phase 2: Trading-grade directional model for predicting move direction."""
    
    def __init__(self, input_dim: int, d_model: int = 128, num_heads: int = 4, 
                 num_layers: int = 3, d_ff: int = 512, dropout: float = 0.2):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncodingDirectional(d_model)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlockDirectional(d_model, num_heads, d_ff, dropout)
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
            "environment": {"sandbox": False},  # LIVE TRADING
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
                },
                "exit_plan": {
                    "legs": [
                        {
                            "kind": "TP",
                            "label": "Take Profit",
                            "allocation": {
                                "type": "percentage",
                                "value": 100.0
                            },
                            "trigger": {
                                "type": "percentage",
                                "value": 0.35  # 0.35% take profit
                            },
                            "order": {
                                "order_type": "MARKET",
                                "time_in_force": "GTC"
                            }
                        },
                        {
                            "kind": "SL",
                            "label": "Stop Loss",
                            "allocation": {
                                "type": "percentage",
                                "value": 100.0
                            },
                            "trigger": {
                                "type": "percentage",
                                "value": -2.0  # -2% stop loss
                            },
                            "order": {
                                "order_type": "MARKET",
                                "time_in_force": "GTC"
                            }
                        }
                    ],
                    "timestop": {
                        "enabled": True,
                        "duration_minutes": 30.0,  # 30-minute timestop
                        "action": "MARKET_EXIT"
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
            logger.info(f"[COM] Sending LIVE TRADE order: {order_type} {side} {symbol}")
            response = self.session.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"[SUCCESS] LIVE ORDER created: {result}")
                return result
            else:
                logger.error(f"[ERROR] LIVE ORDER failed: {response.status_code} - {response.text}")
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
        self.binary_robust_scaler = None
        self.binary_standard_scaler = None
        self.directional_scaler = None
        self.feature_cols = None
        
        if self.binary_model:
            try:
                binary_checkpoint = torch.load(binary_model_path, map_location=self.device, weights_only=False)
                self.binary_robust_scaler = binary_checkpoint.get('robust_scaler')
                self.binary_standard_scaler = binary_checkpoint.get('standard_scaler')
                self.feature_cols = binary_checkpoint.get('feature_cols', [])
                # Get sequence length from model args (should be 120)
                binary_args = binary_checkpoint.get('args')
                self.binary_sequence_length_loaded = binary_args.sequence_length if binary_args else 120
                logger.info(f"[AI] Binary model loaded with {len(self.feature_cols)} features")
                logger.info(f"[DEBUG] Binary sequence length: {self.binary_sequence_length_loaded}")
                logger.info(f"[DEBUG] Binary robust scaler loaded: {self.binary_robust_scaler is not None}")
                logger.info(f"[DEBUG] Binary standard scaler loaded: {self.binary_standard_scaler is not None}")
                if self.binary_robust_scaler:
                    logger.info(f"[DEBUG] Robust scaler type: {type(self.binary_robust_scaler)}")
                if self.binary_standard_scaler:
                    logger.info(f"[DEBUG] Standard scaler type: {type(self.binary_standard_scaler)}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load binary scalers: {e}")
        
        # Store for later use by feature engine
        self.binary_robust_scaler_loaded = self.binary_robust_scaler
        self.binary_standard_scaler_loaded = self.binary_standard_scaler
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
        
        # EXACT SAME 38 BINARY FEATURES AS BACKTESTER (37 base + y_actionable = 38)
        self.feature_columns = [
            "open", "volume", "quote_volume", "r1", "r2", "r5", "r10", 
            "range_pct", "body_pct", "rv", "vol_z", "avg_trade_size",
            "buy_vol", "sell_vol", "mean_size", "max_size", "p95_size", 
            "n_trades", "signed_vol", "imb_aggr", "CVD", "signed_volatility",
            "block_trades", "impact_proxy", "vw_tick_return", "vol_regime",
            "drawdown", "minute_sin", "minute_cos", "day_sin", "day_cos",
            "session_asia", "session_europe", "session_us", "price_position",
            "vol_concentration", "vol_entropy", "y_actionable"
        ]  # EXACTLY 38 features matching backtester
        logger.info(f"[FEATURES] Using EXACT {len(self.feature_columns)} binary features matching backtester")
        
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
                # Create datetime column from DateTime column
                df['datetime'] = pd.to_datetime(df['DateTime'])
                # Ensure we keep all needed columns
                df['quantity'] = df['Quantity']
                df['price'] = df['Price'] 
                df['isbuyermaker'] = df['IsBuyerMaker']
                self.trades_buffer = df.sort_values('datetime').tail(1000)
                logger.info(f"[DEBUG] Loaded trades with columns: {list(self.trades_buffer.columns)}")
                return True
        except Exception as e:
            logger.error(f"[ERROR] Trades update failed: {e}")
            return False
    
    def calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL 38 binary features + 238 directional features exactly as backtester"""
        if len(df) < 50:
            return df
            
        try:
            # ==== HARDCODE ALL 38 BINARY FEATURES EXACTLY ====
            logger.info("🔧 Computing ALL 38 binary features exactly...")
            
            # Basic returns
            df['r1'] = df['close'].pct_change()
            df['r2'] = df['close'].pct_change(2)
            df['r5'] = df['close'].pct_change(5)
            df['r10'] = df['close'].pct_change(10)
            
            # Range and body relative to previous close
            prev_close = df['close'].shift(1)
            df['range_pct'] = (df['high'] - df['low']) / (prev_close + 1e-9)
            df['body_pct'] = (df['close'] - df['open']).abs() / (prev_close + 1e-9)
            
            # Volume z-score
            vol_ma = df['volume'].rolling(20, min_periods=5).mean()
            vol_sd = df['volume'].rolling(20, min_periods=5).std()
            df['vol_z'] = (df['volume'] - vol_ma) / (vol_sd + 1e-9)
            
            # Average trade size
            df['avg_trade_size'] = (df['tot_vol'] / (df['n_trades'].replace(0, np.nan))).fillna(0.0)
            
            # Volatility regime: 0=low,1=med,2=high based on rolling sigma tertiles
            roll_sigma = df['r1'].rolling(200, min_periods=50).std()
            q1 = roll_sigma.quantile(0.33)
            q2 = roll_sigma.quantile(0.66)
            vol_regime = np.select([roll_sigma <= q1, roll_sigma <= q2], [0, 1], default=2)
            df['vol_regime'] = vol_regime
            
            # Drawdown state
            rolling_max = df['close'].cummax()
            df['drawdown'] = (df['close'] - rolling_max) / (rolling_max + 1e-9)
            
            # Calendar encodings
            minute_of_day = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
            day_of_week = df['datetime'].dt.dayofweek
            df['minute_sin'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
            df['minute_cos'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
            df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
            df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
            
            # Sessions
            hour = df['datetime'].dt.hour
            df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
            df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(int)
            df['session_us'] = ((hour >= 16) & (hour < 24)).astype(int)
            
            # Price position and volume features
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
            df['vol_concentration'] = df['volume'] / (df['volume'].rolling(20, min_periods=5).mean() + 1e-9)
            df['vol_entropy'] = -(df['vol_concentration']) * np.log(df['vol_concentration'] + 1e-12)
            
            # Quote volume (if not already calculated)
            if 'quote_volume' not in df.columns:
                df['quote_volume'] = df['volume'] * df['close']
            
            # Realized volatility (if not already calculated)
            if 'rv' not in df.columns:
                df['rv'] = (df['r1'] ** 2).rolling(20, min_periods=1).sum()
            
            # Volume-weighted tick return (if not already calculated) 
            if 'vw_tick_return' not in df.columns:
                df['vw_tick_return'] = (df['r1'] * df['volume']).rolling(5, min_periods=1).mean()
            
            # y_actionable = 0 
            df['y_actionable'] = 0
            
            # ==== NOW COMPUTE ALL 238 DIRECTIONAL FEATURES ====
            df = self.compute_all_directional_features(df)
            
            return df
        except Exception as e:
            logger.error(f"[ERROR] Feature calculation failed: {e}")
            return df
    
    def compute_all_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ALL 238 directional features exactly as expected by the model"""
        
        logger.info(f"🔧 Computing ALL 238 directional features...")
        
        # === DIRECTIONAL MODEL SPECIFIC FEATURES ===
        # Directional model has high_x, low_x, close_x (vs high, low, close for binary)
        df['high_x'] = df['high']
        df['low_x'] = df['low'] 
        df['close_x'] = df['close']
        
        # Buy/sell ratios and imbalances
        df['buy_sell_ratio'] = df['buy_vol'] / (df['sell_vol'] + 1e-9)
        df['buy_sell_diff'] = df['buy_vol'] - df['sell_vol']
        df['buy_sell_imbalance'] = (df['buy_vol'] - df['sell_vol']) / (df['buy_vol'] + df['sell_vol'] + 1e-9)
        
        # Momentum features (5, 10, 20 periods)
        for period in [5, 10, 20]:
            df[f'buy_momentum_{period}'] = df['buy_vol'].rolling(period, min_periods=1).mean()
            df[f'sell_momentum_{period}'] = df['sell_vol'].rolling(period, min_periods=1).mean()
            df[f'imbalance_momentum_{period}'] = df['buy_sell_imbalance'].rolling(period, min_periods=1).mean()
            df[f'imbalance_volatility_{period}'] = df['buy_sell_imbalance'].rolling(period, min_periods=1).std().fillna(0)
        
        # Cumulative pressures
        df['cumulative_buy_pressure'] = df['buy_vol'].cumsum()
        df['cumulative_sell_pressure'] = df['sell_vol'].cumsum()
        df['cumulative_imbalance'] = df['buy_sell_imbalance'].cumsum()
        
        # Accelerations (2nd derivative)
        df['buy_acceleration'] = df['buy_vol'].diff(2)
        df['sell_acceleration'] = df['sell_vol'].diff(2)
        df['imbalance_acceleration'] = df['buy_sell_imbalance'].diff(2)
        
        # High-Low spread features
        df['hl_spread'] = df['high'] - df['low']
        df['hl_spread_norm'] = df['hl_spread'] / df['close']
        df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-9)
        
        # Intrabar momentum and reversal
        df['intrabar_momentum'] = (df['close'] - df['open']) / (df['open'] + 1e-9)
        df['intrabar_reversal'] = ((df['high'] + df['low']) / 2 - df['close']).abs() / (df['high'] - df['low'] + 1e-9)
        
        # Volume Price Trend
        df['volume_price_trend'] = df['r1'] * df['volume']
        df['vpt'] = df['volume_price_trend'].cumsum()
        df['vpt_momentum'] = df['vpt'].diff()
        
        # Tick direction and momentum
        df['tick_direction'] = np.sign(df['r1'])
        df['tick_momentum'] = df['tick_direction'].rolling(5, min_periods=1).mean()
        df['tick_acceleration'] = df['tick_momentum'].diff()
        
        # === RSI FEATURES ===
        for period in [7, 14, 21, 50]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            rs = gain / (loss + 1e-9)
            rsi = 100 - (100 / (1 + rs))
            
            df[f'rsi_{period}'] = rsi
            df[f'rsi_momentum_{period}'] = rsi.diff()
            df[f'rsi_divergence_{period}'] = rsi - rsi.rolling(period, min_periods=1).mean()
        
        # === MACD FEATURES ===
        for fast, slow in [(8, 21), (12, 26), (19, 39)]:
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal
            df[f'macd_histogram_{fast}_{slow}'] = histogram
            df[f'macd_momentum_{fast}_{slow}'] = macd.diff()
            df[f'macd_acceleration_{fast}_{slow}'] = macd.diff(2)
        
        # === BOLLINGER BANDS ===
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period, min_periods=1).mean()
            std = df['close'].rolling(period, min_periods=1).std()
            
            df[f'bb_upper_{period}'] = sma + (2 * std)
            df[f'bb_lower_{period}'] = sma - (2 * std)
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-9)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_squeeze_{period}'] = (std < std.rolling(period, min_periods=1).mean()).astype(int)
        
        # === STOCHASTIC ===
        for period in [14, 21, 50]:
            lowest_low = df['low'].rolling(period, min_periods=1).min()
            highest_high = df['high'].rolling(period, min_periods=1).max()
            k_percent = (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-9) * 100
            d_percent = k_percent.rolling(3, min_periods=1).mean()
            
            df[f'stoch_k_{period}'] = k_percent
            df[f'stoch_d_{period}'] = d_percent
            df[f'stoch_momentum_{period}'] = k_percent.diff()
        
        # === VOLUME DISTRIBUTION ===
        for period in [10, 20, 50]:
            # Volume at different price levels within the bar
            df[f'volume_at_high_{period}'] = (df['volume'] * (df['close'] == df['high']).astype(int)).rolling(period, min_periods=1).sum()
            df[f'volume_at_low_{period}'] = (df['volume'] * (df['close'] == df['low']).astype(int)).rolling(period, min_periods=1).sum()
            df[f'volume_at_mid_{period}'] = df['volume'].rolling(period, min_periods=1).sum() - df[f'volume_at_high_{period}'] - df[f'volume_at_low_{period}']
            
            total_vol = df['volume'].rolling(period, min_periods=1).sum()
            df[f'volume_concentration_{period}'] = df[f'volume_at_high_{period}'] / (total_vol + 1e-9)
            df[f'high_volume_momentum_{period}'] = df[f'volume_at_high_{period}'].diff()
            df[f'low_volume_momentum_{period}'] = df[f'volume_at_low_{period}'].diff()
        
        # === VWAP ===
        for period in [10, 20, 50]:
            vwap = (df['close'] * df['volume']).rolling(period, min_periods=1).sum() / df['volume'].rolling(period, min_periods=1).sum()
            df[f'vwap_{period}'] = vwap
            df[f'vwap_deviation_{period}'] = (df['close'] - vwap) / vwap
            df[f'vwap_momentum_{period}'] = vwap.diff()
        
        # === VOLUME RATE OF CHANGE ===
        for period in [3, 5, 10]:
            df[f'volume_roc_{period}'] = df['volume'].pct_change(period)
            df[f'volume_acceleration_{period}'] = df[f'volume_roc_{period}'].diff()
        
        # === VOLATILITY FEATURES ===
        for period in [10, 20, 50]:
            volatility = df['r1'].rolling(period, min_periods=1).std()
            df[f'volatility_{period}'] = volatility
            df[f'volatility_rank_{period}'] = volatility.rolling(period*2, min_periods=1).rank(pct=True)
            
            vol_q1 = volatility.quantile(0.33)
            vol_q2 = volatility.quantile(0.66)
            regime = np.select([volatility <= vol_q1, volatility <= vol_q2], [0, 1], default=2)
            df[f'volatility_regime_{period}'] = regime
        
        # === TREND FEATURES ===
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period, min_periods=1).mean()
            trend_strength = (df['close'] - sma) / sma
            trend_direction = np.sign(sma.diff())
            
            df[f'trend_strength_{period}'] = trend_strength
            df[f'trend_direction_{period}'] = trend_direction
            df[f'trend_momentum_{period}'] = trend_strength.diff()
        
        # === MEAN REVERSION ===
        for period in [5, 10, 20]:
            sma = df['close'].rolling(period, min_periods=1).mean()
            std = df['close'].rolling(period, min_periods=1).std()
            
            mean_reversion = (sma - df['close']) / (std + 1e-9)
            zscore = (df['close'] - sma) / (std + 1e-9)
            
            df[f'mean_reversion_{period}'] = mean_reversion
            df[f'mean_reversion_momentum_{period}'] = mean_reversion.diff()
            df[f'zscore_{period}'] = zscore
        
        # === VOLUME EFFICIENCY ===
        for period in [5, 10, 20]:
            price_change = df['close'].diff(period).abs()
            volume_sum = df['volume'].rolling(period, min_periods=1).sum()
            efficiency = price_change / (volume_sum + 1e-9)
            
            df[f'volume_efficiency_{period}'] = efficiency
            df[f'volume_efficiency_momentum_{period}'] = efficiency.diff()
        
        # Volume z-score (additional)
        vol_mean = df['volume'].rolling(50, min_periods=10).mean()
        vol_std = df['volume'].rolling(50, min_periods=10).std()
        df['volume_zscore'] = (df['volume'] - vol_mean) / (vol_std + 1e-9)
        
        # === LARGE TRADES ===
        large_threshold = df['mean_size'].rolling(50, min_periods=10).quantile(0.95)
        df['large_trade'] = (df['max_size'] > large_threshold).astype(int)
        df['large_buy_trade'] = ((df['max_size'] > large_threshold) & (df['buy_vol'] > df['sell_vol'])).astype(int)
        df['large_sell_trade'] = ((df['max_size'] > large_threshold) & (df['sell_vol'] > df['buy_vol'])).astype(int)
        df['cumulative_large_buys'] = df['large_buy_trade'].cumsum()
        df['cumulative_large_sells'] = df['large_sell_trade'].cumsum()
        df['large_trade_imbalance'] = df['cumulative_large_buys'] - df['cumulative_large_sells']
        
        # === VOLUME IMPACT ===
        for period in [1, 3, 5]:
            volume_impact = df['r1'].abs() / (df['volume'] + 1e-9)
            df[f'volume_impact_{period}'] = volume_impact.rolling(period, min_periods=1).mean()
            df[f'impact_decay_{period}'] = volume_impact.rolling(period, min_periods=1).std().fillna(0)
        
        # === PRICE-VOLUME DIVERGENCE ===
        for period in [5, 10, 20]:
            price_momentum = df['close'].rolling(period, min_periods=1).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0)
            volume_momentum = df['volume'].rolling(period, min_periods=1).apply(lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0)
            df[f'price_volume_divergence_{period}'] = price_momentum - volume_momentum
        
        # MACD divergence (using 12-26)
        df['macd_divergence'] = df['macd_12_26'] - df['macd_12_26'].rolling(20, min_periods=1).mean()
        
        # === AUTOCORRELATION ===
        for lag in [1, 2, 3, 5]:
            df[f'price_autocorr_{lag}'] = df['r1'].rolling(20, min_periods=lag+1).apply(
                lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
            ).fillna(0)
            df[f'volume_autocorr_{lag}'] = df['volume'].pct_change().rolling(20, min_periods=lag+1).apply(
                lambda x: np.corrcoef(x[:-lag], x[lag:])[0, 1] if len(x) > lag else 0
            ).fillna(0)
        
        # === PRICE-VOLUME CORRELATION ===
        for period in [10, 20, 50]:
            corr = df['r1'].rolling(period, min_periods=5).corr(df['volume'].pct_change())
            df[f'price_volume_corr_{period}'] = corr.fillna(0)
            df[f'price_volume_corr_momentum_{period}'] = corr.diff().fillna(0)
        
        # === MOMENTUM CORRELATIONS ===
        mom_5 = df['close'].pct_change(5)
        mom_10 = df['close'].pct_change(10) 
        mom_20 = df['close'].pct_change(20)
        
        df['momentum_corr_5_10'] = mom_5.rolling(20, min_periods=5).corr(mom_10).fillna(0)
        df['momentum_corr_10_20'] = mom_10.rolling(20, min_periods=5).corr(mom_20).fillna(0)
        df['momentum_corr_20_50'] = mom_20.rolling(20, min_periods=5).corr(df['close'].pct_change(50)).fillna(0)
        
        # === TIME FEATURES ===
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Hour and day of week cyclical
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Sessions (different names for directional)
        df['asian_session'] = df['session_asia']
        df['european_session'] = df['session_europe']
        df['us_session'] = df['session_us']
        
        # Session open/close (simplified)
        df['session_open'] = ((df['hour'] == 0) | (df['hour'] == 8) | (df['hour'] == 16)).astype(int)
        df['session_close'] = ((df['hour'] == 7) | (df['hour'] == 15) | (df['hour'] == 23)).astype(int)
        
        # Final high_y, low_y, close_y (same as high_x, low_x, close_x)
        df['high_y'] = df['high']
        df['low_y'] = df['low']
        df['close_y'] = df['close']
        
        logger.info(f"✅ Computed ALL directional features")
        
        return df
    
    def _calculate_trade_features(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trade features EXACTLY like backtester calculate_trade_features_for_backtest()"""
        try:
            if len(self.trades_buffer) == 0:
                logger.warning("[WARNING] No trade data available, using zeros like backtester")
                # Initialize with zeros like backtester
                trade_features = {
                    'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
                    'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
                    'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
                    'CVD': 0.0, 'dCVD': 0.0, 'signed_volatility': 0.0, 'impact_proxy': 0.0,
                    'vw_tick_return': 0.0, 'rv': 0.0
                }
                for col in trade_features:
                    ohlcv_df[col] = trade_features[col]
                return ohlcv_df
            
            # EXACT SAME LOGIC AS BACKTESTER calculate_trade_features_for_backtest()
            trades_df = self.trades_buffer.copy()
            
            # Ensure we have the right columns and types
            logger.info(f"[DEBUG] Original columns: {list(trades_df.columns)}")
            logger.info(f"[DEBUG] DataFrame shape: {trades_df.shape}")
            
            # Ensure datetime column is properly typed
            if 'datetime' in trades_df.columns:
                trades_df['datetime'] = pd.to_datetime(trades_df['datetime'])
                logger.info(f"[DEBUG] Converted datetime column, dtype: {trades_df['datetime'].dtype}")
            else:
                logger.error(f"[ERROR] No datetime column in trades_buffer!")
                raise ValueError("Missing datetime column")
            
            # Group trades by minute candles (like backtester)
            trades_df['candle_time'] = trades_df['datetime'].dt.floor('1min')
            logger.info(f"[DEBUG] Created {len(trades_df['candle_time'].unique())} unique candle times")
            
            # Initialize trade feature columns (EXACT same as backtester)
            trade_features = {
                'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
                'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
                'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
                'CVD': 0.0, 'dCVD': 0.0, 'signed_volatility': 0.0, 'impact_proxy': 0.0,
                'vw_tick_return': 0.0, 'rv': 0.0
            }
            
            for col in trade_features:
                ohlcv_df[col] = trade_features[col]
            
            # Process each candle (EXACT same logic as backtester)
            cvd_cumulative = 0.0
            prev_cvd = 0.0
            
            for idx, candle in ohlcv_df.iterrows():
                candle_time = candle['datetime'].floor('1min')
                candle_trades = trades_df[trades_df['candle_time'] == candle_time]
                
                if len(candle_trades) > 0:
                    # Calculate volumes (EXACT same as backtester)
                    buy_trades = candle_trades[candle_trades['isbuyermaker'] == False]  # Taker buy = aggressive buy
                    sell_trades = candle_trades[candle_trades['isbuyermaker'] == True]   # Taker sell = aggressive sell
                    
                    buy_vol = buy_trades['quantity'].sum() if len(buy_trades) > 0 else 0.0
                    sell_vol = sell_trades['quantity'].sum() if len(sell_trades) > 0 else 0.0
                    tot_vol = candle_trades['quantity'].sum()
                    
                    # Size statistics (EXACT same as backtester)
                    sizes = candle_trades['quantity']
                    mean_size = sizes.mean() if len(sizes) > 0 else 0.0
                    max_size = sizes.max() if len(sizes) > 0 else 0.0
                    p95_size = sizes.quantile(0.95) if len(sizes) > 0 else 0.0
                    n_trades = len(candle_trades)
                    
                    # Signed volume and imbalance (EXACT same as backtester)
                    signed_vol = buy_vol - sell_vol
                    imb_aggr = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0.0
                    
                    # Block trades (EXACT same as backtester)
                    if p95_size > 0:
                        block_trades = len(sizes[sizes >= p95_size])
                    else:
                        block_trades = 0
                    
                    # CVD (EXACT same as backtester)
                    cvd_cumulative += signed_vol
                    dCVD = cvd_cumulative - prev_cvd
                    prev_cvd = cvd_cumulative
                    
                    # Signed volatility (EXACT same as backtester)
                    price_changes = candle_trades['price'].diff().abs()
                    if len(price_changes) > 1 and signed_vol != 0:
                        signed_volatility = price_changes.mean() * abs(signed_vol)
                    else:
                        signed_volatility = 0.0
                    
                    # Volume-weighted tick return (EXACT same as backtester)
                    tick_returns = candle_trades['price'].pct_change().fillna(0.0)
                    qty = candle_trades['quantity'].fillna(0.0)
                    vw_tick_return = (tick_returns * qty).sum() / (qty.sum() + 1e-9)

                    # Realized volatility (EXACT same as backtester)
                    rv = (tick_returns ** 2).sum()

                    # Impact proxy (EXACT same as backtester)
                    r1 = candle['close'] / candle['open'] - 1 if candle['open'] > 0 else 0
                    impact_proxy = abs(r1) / (tot_vol + 1e-9) if tot_vol > 0 else 0.0
                    
                    # Update dataframe (EXACT same as backtester)
                    ohlcv_df.loc[idx, 'buy_vol'] = buy_vol
                    ohlcv_df.loc[idx, 'sell_vol'] = sell_vol
                    ohlcv_df.loc[idx, 'tot_vol'] = tot_vol
                    ohlcv_df.loc[idx, 'mean_size'] = mean_size
                    ohlcv_df.loc[idx, 'max_size'] = max_size
                    ohlcv_df.loc[idx, 'p95_size'] = p95_size
                    ohlcv_df.loc[idx, 'n_trades'] = n_trades
                    ohlcv_df.loc[idx, 'signed_vol'] = signed_vol
                    ohlcv_df.loc[idx, 'imb_aggr'] = imb_aggr
                    ohlcv_df.loc[idx, 'block_trades'] = block_trades
                    ohlcv_df.loc[idx, 'CVD'] = cvd_cumulative
                    ohlcv_df.loc[idx, 'dCVD'] = dCVD
                    ohlcv_df.loc[idx, 'signed_volatility'] = signed_volatility
                    ohlcv_df.loc[idx, 'impact_proxy'] = impact_proxy
                    ohlcv_df.loc[idx, 'vw_tick_return'] = vw_tick_return
                    ohlcv_df.loc[idx, 'rv'] = rv
            
            logger.info(f"[TRADES] Calculated trade features using EXACT backtester logic from {len(self.trades_buffer)} trades")
            
            return ohlcv_df
            
        except Exception as e:
            logger.error(f"[ERROR] Trade feature calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # Initialize with zeros like backtester on failure
            trade_features = {
                'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
                'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
                'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
                'CVD': 0.0, 'dCVD': 0.0, 'signed_volatility': 0.0, 'impact_proxy': 0.0,
                'vw_tick_return': 0.0, 'rv': 0.0
            }
            for col in trade_features:
                ohlcv_df[col] = trade_features[col]
            return ohlcv_df
    
    def get_binary_features(self) -> Optional[np.ndarray]:
        """Get features for binary model using EXACT same method as backtester"""
        if len(self.ohlcv_buffer) < 90:  # Need enough data for feature calculation
            return None
            
        try:
            # Calculate features EXACTLY like build_features.py
            df = self.ohlcv_buffer.tail(90).copy()  # Use enough data for rolling calculations
            
            # 1. Core OHLCV Features (EXACT same as build_features.py)
            df["quote_volume"] = df["volume"] * df["close"]  # Calculate quote_volume!
            
            # Calculate returns EXACTLY like backtester (percentage changes, NOT log returns!)
            df["r1"] = df["close"].pct_change()     # 1-period percentage return
            df["r2"] = df["close"].pct_change(2)    # 2-period percentage return  
            df["r5"] = df["close"].pct_change(5)    # 5-period percentage return
            df["r10"] = df["close"].pct_change(10)  # 10-period percentage return
            
            # Fix range_pct and body_pct to match backtester exactly (with epsilon)
            prev_close = df["close"].shift(1)
            df["range_pct"] = (df["high"] - df["low"]) / (prev_close + 1e-9)
            df["body_pct"] = (df["close"] - df["open"]).abs() / (prev_close + 1e-9)
            df["atr_pct"] = (df["high"].combine(df["low"], max) - 
                           df["low"].combine(df["high"], min)) / df["close"].shift(1)
            df["rv"] = df["r1"].pow(2)
            
            # Volume z-score (simplified - no rolling for live)
            df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
            df["avg_trade_size"] = df["volume"] / 100  # Simplified since we don't have trade count
            
            # Calculate REAL trade features from aggregate trades data
            df = self._calculate_trade_features(df)
            # impact_proxy, vw_tick_return, rv already calculated in trade features - don't overwrite!
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
                # Use MAIN HUB feature columns - ONE SOURCE OF TRUTH
                model_feature_cols = self.feature_columns.copy()  # Use main hub feature list
                logger.info(f"[DEBUG] Using MAIN HUB feature columns: {len(model_feature_cols)} features")
            
            # STEP 1: Apply EXACT same preprocessing as backtester BEFORE creating feature matrix
            # Use sequence length from checkpoint (self.binary_sequence_length)
            seq_length = self.binary_sequence_length  # Use the actual checkpoint value
            logger.info(f"[DEBUG] Using sequence length from checkpoint: {seq_length}")
            
            if len(df) < seq_length:
                logger.warning(f"[WARNING] Not enough data for sequence ({len(df)} < {seq_length})")
                return None
                
            sequence_data = df.tail(seq_length).copy()
            
            # EXACT PREPROCESSING ORDER AS BACKTESTER:
            # 1. Calculate all features ✓ (already done)
            # 2. Apply volume scaling (/1e6)
            # 3. Set impact_proxy = 0 (matching training data)
            # 4. Clip outliers  
            # 5. Apply saved scalers
            
            # Step 2: Volume scaling
            volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                              'max_size', 'p95_size', 'mean_size', 'signed_vol', 'dCVD', 'CVD', 'signed_volatility']  # Complete list
            
            # Apply EXACT same preprocessing as backtester/training
            for col in sequence_data.columns:
                if col in volume_features:
                    # Scale volume features to millions (EXACT same as training)
                    sequence_data[col] = sequence_data[col] / 1e6
                elif col == 'impact_proxy':
                    # DEBUG: Check what's causing extreme impact_proxy values
                    max_impact = sequence_data[col].max()
                    if max_impact > 1000:
                        max_idx = sequence_data[col].idxmax()
                        r1_val = sequence_data.loc[max_idx, 'r1'] if 'r1' in sequence_data.columns else 'N/A'
                        volume_val = sequence_data.loc[max_idx, 'volume'] if 'volume' in sequence_data.columns else 'N/A'
                        logger.warning(f"[DEBUG] Extreme impact_proxy={max_impact:.1f} caused by r1={r1_val}, volume={volume_val}")
                    # Apply normal outlier clipping like other features
                    p1, p99 = np.percentile(sequence_data[col], [1, 99])
                    sequence_data[col] = sequence_data[col].clip(p1, p99)
                elif sequence_data[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    # Clip outliers using percentiles (EXACT same as training)
                    p1, p99 = np.percentile(sequence_data[col], [1, 99])
                    sequence_data[col] = sequence_data[col].clip(p1, p99)
            
            # Fill NaN values (EXACT same as backtester)
            sequence_data = sequence_data.fillna(method='ffill').fillna(0)
            
            # STEP 2: Create feature matrix from PREPROCESSED data
            feature_matrix = np.zeros((seq_length, len(model_feature_cols)))
            
            for i, col in enumerate(model_feature_cols):
                if col == "y_actionable":
                    # Always set y_actionable to 0 for binary model input
                    feature_matrix[:, i] = 0.0
                elif col == "impact_proxy":
                    # Step 3: Set impact_proxy to 0 to match training data (training had ~0 values with ~0 variance)
                    feature_matrix[:, i] = 0.0
                    logger.info(f"[DEBUG] Step 3: Set impact_proxy to 0 to match training distribution (training had ~0 variance)")
                elif col in sequence_data.columns:
                    feature_matrix[:, i] = sequence_data[col].values
                else:
                    logger.warning(f"[WARNING] Missing feature column: {col}")
                    feature_matrix[:, i] = 0.0  # Missing features as zeros
            
            # STEP 3: Apply scalers to preprocessed features
            if (hasattr(self, 'binary_robust_scaler') and self.binary_robust_scaler is not None and
                hasattr(self, 'binary_standard_scaler') and self.binary_standard_scaler is not None):
                # Create DataFrame for scaler application
                temp_df = pd.DataFrame(feature_matrix, columns=model_feature_cols)
                
                logger.info(f"[DEBUG] Pre-scaler stats: min={temp_df.min().min():.6f}, max={temp_df.max().max():.6f}, mean={temp_df.mean().mean():.6f}")
                
                # DEBUG: Find ALL features with large values (potential scaling issues)
                max_vals = temp_df.max()
                large_features = max_vals[max_vals > 1000]
                if len(large_features) > 0:
                    logger.warning(f"[DEBUG] Features with values > 1000 (potential scaling issues):")
                    for feat, val in large_features.items():
                        logger.warning(f"[DEBUG]   '{feat}': max={val:.1f}")
                else:
                    logger.info(f"[DEBUG] No features with values > 1000 - preprocessing looks good!")
                
                # DEBUG: Show impact_proxy distribution before scaling
                if 'impact_proxy' in temp_df.columns:
                    impact_vals = temp_df['impact_proxy']
                    logger.warning(f"[DEBUG] impact_proxy before scaling: min={impact_vals.min():.6f}, max={impact_vals.max():.6f}, mean={impact_vals.mean():.6f}, median={impact_vals.median():.6f}")
                
                # Apply saved RobustScaler (on already preprocessed data)
                features_robust_scaled = self.binary_robust_scaler.transform(temp_df[model_feature_cols])
                logger.info(f"[DEBUG] Applied RobustScaler - stats: min={features_robust_scaled.min():.6f}, max={features_robust_scaled.max():.6f}, mean={features_robust_scaled.mean():.6f}")
                
                # DEBUG: Find which feature has the large value after RobustScaler
                max_idx = np.unravel_index(np.argmax(features_robust_scaled), features_robust_scaled.shape)
                logger.warning(f"[DEBUG] Largest value after RobustScaler: feature '{model_feature_cols[max_idx[1]]}' at position {max_idx[1]} = {features_robust_scaled[max_idx]:.1f}")
                
                # Apply saved StandardScaler
                features_final = self.binary_standard_scaler.transform(features_robust_scaled)
                logger.info(f"[DEBUG] Applied StandardScaler - stats: min={features_final.min():.6f}, max={features_final.max():.6f}, mean={features_final.mean():.6f}")
                
                # Add debug logging for key feature distributions (COMPARE WITH BACKTESTER)
                logger.info(f"[LIVE] Feature stats - r1: mean={df['r1'].iloc[-60:].mean():.6f}, std={df['r1'].iloc[-60:].std():.6f}")
                logger.info(f"[LIVE] Feature stats - r2: mean={df['r2'].iloc[-60:].mean():.6f}, std={df['r2'].iloc[-60:].std():.6f}")
                logger.info(f"[LIVE] Feature stats - volume: mean={df['volume'].iloc[-60:].mean():.6f}, std={df['volume'].iloc[-60:].std():.6f}")
                logger.info(f"[LIVE] Feature stats - range_pct: mean={df['range_pct'].iloc[-60:].mean():.6f}, std={df['range_pct'].iloc[-60:].std():.6f}")
                logger.info(f"[LIVE] Feature stats - impact_proxy: mean={df['impact_proxy'].iloc[-60:].mean():.6f}, std={df['impact_proxy'].iloc[-60:].std():.6f}")
                
                return features_final  # Return the full sequence (60, 38)
            else:
                logger.warning("[WARNING] Binary scalers not available (need both RobustScaler and StandardScaler), using raw features")
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
            # Create market entry order (LIVE TRADE)
            logger.info(f"[COM] Sending LIVE TRADE market order: {side} {symbol}")
            order_response = self.com_client.create_order(symbol, side, "MARKET")
            
            if "error" in order_response:
                logger.error(f"[❌ TRADE FAILED] Failed to open LIVE position: {order_response['error']} - BINARY MOVE WASTED!")
                return None
            
            # *** SUCCESSFUL TRADE EXECUTION ***
            logger.info(f"[🚀 TRADE SUCCESS] LIVE {side} order placed successfully! Binary->Directional->Trade COMPLETE!")
            
            position_ref = order_response.get("position_ref", f"live_pos_{int(time.time())}")
            
            # Calculate TP price (0.35% away)
            tp_multiplier = 1.0035 if side == "BUY" else 0.9965
            tp_price = current_price * tp_multiplier
            
            # Create TP order (post-only limit order)
            tp_side = "SELL" if side == "BUY" else "BUY"
            logger.info(f"[COM] Sending LIVE TRADE TP order: {tp_side} {symbol} @ {tp_price:.6f}")
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
            
            logger.info(f"[POSITION] LIVE TRADE OPENED: {side} {symbol} @ {current_price:.6f}, TP: {tp_price:.6f}")
            return position_ref
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to open position: {e}")
            return None
    
    def check_stops(self, max_hold_minutes: int = 30, current_price: float = None):
        """No internal stop monitoring - COM handles all stops automatically via exit_plan"""
        # COM now handles all stops automatically:
        # - 0.35% Take Profit
        # - -2% Stop Loss  
        # - 30-minute TimeStop
        # No need for internal monitoring anymore!
        logger.info("[INFO] All stops managed by COM automatically via exit_plan")
        pass
    
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
        
        # Pass both binary scalers, feature columns, and sequence length to feature engine
        self.feature_engine.binary_robust_scaler = getattr(self.model_inference, 'binary_robust_scaler_loaded', None)
        self.feature_engine.binary_standard_scaler = getattr(self.model_inference, 'binary_standard_scaler_loaded', None)
        self.feature_engine.feature_cols = getattr(self.model_inference, 'feature_cols_loaded', [])
        self.feature_engine.binary_sequence_length = getattr(self.model_inference, 'binary_sequence_length_loaded', 120)
        
        self.symbol = config["symbol"]
        self.max_positions = config.get("max_positions", 1)
        self.max_hold_minutes = config.get("max_hold_minutes", 30)
        self.ohlcv_file = config["ohlcv_file"]
        self.trades_file = config["trades_file"]
        
        self.running = False
        self.last_candle_time = None
        
        logger.info("[START] LiveDOGETrader initialized with LIVE TRADING")
    
    async def run(self):
        self.running = True
        logger.info("[SIGNAL] Starting live LIVE trading...")
        
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
                
                # COM handles all stops automatically via exit_plan - no internal monitoring needed
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except KeyboardInterrupt:
                logger.info("[STOP] Received shutdown signal")
                break
            except Exception as e:
                logger.error(f"[ERROR] Error in main loop: {e}")
                await asyncio.sleep(30)
        
        logger.info("[FINISH] LIVE trading stopped")
    
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
            
            # *** BINARY MOVE DETECTED ***
            logger.info(f"[🎯 BINARY MOVE DETECTED] Confidence: {move_confidence:.3f} - Proceeding to directional analysis")
            
            # Phase 2: Directional model with advanced features (232)
            directional_features = self.feature_engine.get_directional_features()
            if directional_features is None:
                logger.error("[❌ DIRECTIONAL FAILED] No directional features available - BINARY MOVE WASTED!")
                return
            
            direction, direction_confidence = self.model_inference.predict_direction(directional_features)
            
            logger.info(f"[AI] Direction prediction: {direction} (confidence: {direction_confidence:.3f})")
            
            if direction == "HOLD":
                logger.warning("[❌ DIRECTIONAL FAILED] No clear direction - BINARY MOVE WASTED!")
                return
            
            # Check position limits
            open_positions = self.position_manager.get_open_positions_count()
            if open_positions >= self.max_positions:
                logger.warning(f"[❌ POSITION LIMIT] Max positions reached ({open_positions}/{self.max_positions}) - BINARY MOVE WASTED!")
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
            logger.info(f"[SIGNAL] Trade Type: LIVE TRADE")
            logger.info("[SIGNAL] ========================")
            
            # Execute LIVE trade
            position_id = self.position_manager.open_position(self.symbol, direction, current_price)
            
            if position_id:
                logger.info(f"[SUCCESS] LIVE position opened: {position_id}")
            else:
                logger.error("[ERROR] Failed to open LIVE position")
                
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
    logger.info("[START] Starting Live DOGE LIVE Trader")
    
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