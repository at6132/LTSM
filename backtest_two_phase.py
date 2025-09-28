#!/usr/bin/env python3
"""
Two-Phase Trading System Backtester

Phase 1: Binary model detects when moves will happen (85% accuracy)
Phase 2: Directional model predicts direction of detected moves (57.4% accuracy)

Trading Parameters:
- Account: $100,000
- Position size: 10% of account per trade
- Leverage: 25x
- Fees: 0.03% per trade (0.06% round trip)
- Test period: All 2025 data
"""

import argparse
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
import os
from typing import Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# GUI imports
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import queue
import time

from train_baseline import BinaryMoveModel, BinaryMoveDataset, load_features_from_checkpoint, create_binary_targets
from train_phase2_directional import TradingDirectionalModel, DirectionalDataset, create_directional_targets

# --- Configure logging to both file and console, and tee print output to logger ---
def _configure_logging_once():
    try:
        # Avoid re-configuring if already set
        if getattr(_configure_logging_once, "_configured", False):
            return
        log_path = os.path.join("Live", "DOGE", "backtest_two_phase.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='a', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        # Tee stdout/stderr prints to logger as INFO/ERROR
        import sys
        class _StreamToLogger:
            def __init__(self, logger, level):
                self.logger = logger
                self.level = level
                self._buffer = ""
            def write(self, message):
                if message is None:
                    return
                self._buffer += str(message)
                while "\n" in self._buffer:
                    line, self._buffer = self._buffer.split("\n", 1)
                    line = line.rstrip()
                    if line:
                        self.logger.log(self.level, line)
            def flush(self):
                if self._buffer:
                    msg = self._buffer.rstrip()
                    self._buffer = ""
                    if msg:
                        self.logger.log(self.level, msg)

        logger = logging.getLogger("backtester")
        sys.stdout = _StreamToLogger(logger, logging.INFO)
        sys.stderr = _StreamToLogger(logger, logging.ERROR)

        _configure_logging_once._configured = True
    except Exception:
        # If logging setup fails, continue without failing the run
        pass

_configure_logging_once()

class BacktestGUI:
    def __init__(self):
        print("🖥️  [DEBUG] Initializing BacktestGUI...")
        try:
            self.root = tk.Tk()
            print("🖥️  [DEBUG] Tkinter root created successfully")
            self.root.title("🚀 Two-Phase Trading System - Live Backtest")
            self.root.geometry("1400x900")
            self.root.configure(bg='#1e1e1e')
            print("🖥️  [DEBUG] Root window configured")
            
            # Data queues for thread-safe updates
            self.update_queue = queue.Queue()
            
            # Initialize data
            self.equity_data = []
            self.time_data = []
            self.trades_data = []
            
            print("🖥️  [DEBUG] Setting up GUI components...")
            self.setup_gui()
            print("🖥️  [DEBUG] GUI setup complete")
        except Exception as e:
            print(f"🖥️  [ERROR] Failed to initialize GUI: {e}")
            import traceback
            traceback.print_exc()
        
    def setup_gui(self):
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top stats panel
        stats_frame = ttk.LabelFrame(main_frame, text="📊 Live Statistics", padding=10)
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Stats grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # Left column stats
        left_stats = ttk.Frame(stats_grid)
        left_stats.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.equity_label = ttk.Label(left_stats, text="💰 Total Equity: $100,000", font=('Arial', 14, 'bold'))
        self.equity_label.pack(anchor=tk.W)
        
        self.pnl_label = ttk.Label(left_stats, text="📈 Total P&L: $0.00", font=('Arial', 12))
        self.pnl_label.pack(anchor=tk.W)
        
        self.fees_label = ttk.Label(left_stats, text="💸 Total Fees: $0.00", font=('Arial', 12))
        self.fees_label.pack(anchor=tk.W)
        
        # Right column stats
        right_stats = ttk.Frame(stats_grid)
        right_stats.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.trades_label = ttk.Label(right_stats, text="🔄 Total Trades: 0", font=('Arial', 12))
        self.trades_label.pack(anchor=tk.E)
        
        self.win_rate_label = ttk.Label(right_stats, text="🎯 Win Rate: 0.0%", font=('Arial', 12))
        self.win_rate_label.pack(anchor=tk.E)
        
        self.position_label = ttk.Label(right_stats, text="📍 Position: NONE", font=('Arial', 12))
        self.position_label.pack(anchor=tk.E)
        
        # Charts container
        charts_frame = ttk.Frame(main_frame)
        charts_frame.pack(fill=tk.BOTH, expand=True)
        
        # Equity chart
        equity_frame = ttk.LabelFrame(charts_frame, text="📈 Equity Curve")
        equity_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.equity_fig = Figure(figsize=(8, 6), facecolor='#2e2e2e')
        self.equity_ax = self.equity_fig.add_subplot(111, facecolor='#2e2e2e')
        self.equity_ax.set_title("Account Equity Over Time", color='white')
        self.equity_ax.tick_params(colors='white')
        self.equity_canvas = FigureCanvasTkAgg(self.equity_fig, equity_frame)
        self.equity_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Recent trades table
        trades_frame = ttk.LabelFrame(charts_frame, text="📋 Recent Trades")
        trades_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Trades treeview
        columns = ('Time', 'Side', 'Entry', 'Exit', 'P&L', 'Duration', 'Reason')
        self.trades_tree = ttk.Treeview(trades_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=80)
        
        trades_scrollbar = ttk.Scrollbar(trades_frame, orient=tk.VERTICAL, command=self.trades_tree.yview)
        self.trades_tree.configure(yscrollcommand=trades_scrollbar.set)
        
        self.trades_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        trades_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Progress bar at bottom
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(main_frame, text="🚀 Initializing backtest...")
        self.status_label.pack(pady=(5, 0))
        
        # Start update loop
        self.update_display()
        
    def update_display(self):
        """Update GUI with queued data"""
        try:
            while True:
                update_data = self.update_queue.get_nowait()
                
                if update_data['type'] == 'stats':
                    self.update_stats(update_data)
                elif update_data['type'] == 'trade':
                    self.add_trade(update_data)
                elif update_data['type'] == 'equity':
                    self.update_equity_chart(update_data)
                elif update_data['type'] == 'progress':
                    self.update_progress(update_data)
                    
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self.update_display)
    
    def update_stats(self, data):
        """Update statistics labels"""
        self.equity_label.config(text=f"💰 Total Equity: ${data['equity']:,.2f}")
        self.pnl_label.config(text=f"📈 Total P&L: ${data['pnl']:+,.2f}")
        self.fees_label.config(text=f"💸 Total Fees: ${data['fees']:,.2f}")
        self.trades_label.config(text=f"🔄 Total Trades: {data['trades']}")
        self.win_rate_label.config(text=f"🎯 Win Rate: {data['win_rate']:.1f}%")
        self.position_label.config(text=f"📍 Position: {data['position']}")
    
    def add_trade(self, data):
        """Add new trade to the table"""
        self.trades_tree.insert('', 0, values=(
            data['time'], data['side'], f"${data['entry']:.6f}", 
            f"${data['exit']:.6f}", f"${data['pnl']:+,.2f}", 
            f"{data['duration']:.1f}min", data['reason']
        ))
        
        # Keep only last 50 trades
        children = self.trades_tree.get_children()
        if len(children) > 50:
            self.trades_tree.delete(children[-1])
    
    def update_equity_chart(self, data):
        """Update equity curve chart"""
        self.equity_data.append(data['equity'])
        self.time_data.append(data['time'])
        
        # Keep last 1000 points for performance
        if len(self.equity_data) > 1000:
            self.equity_data = self.equity_data[-1000:]
            self.time_data = self.time_data[-1000:]
        
        self.equity_ax.clear()
        self.equity_ax.plot(self.time_data, self.equity_data, color='#00ff88', linewidth=2)
        self.equity_ax.set_title("Account Equity Over Time", color='white')
        self.equity_ax.tick_params(colors='white')
        self.equity_ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        self.equity_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        self.equity_canvas.draw()
    
    def update_progress(self, data):
        """Update progress bar and status"""
        self.progress_var.set(data['progress'])
        self.status_label.config(text=data['status'])
    
    def queue_update(self, update_data):
        """Thread-safe way to queue GUI updates"""
        self.update_queue.put(update_data)
    
    def run(self):
        """Start the GUI main loop"""
        print("🖥️  [DEBUG] Starting GUI mainloop...")
        try:
            self.root.mainloop()
            print("🖥️  [DEBUG] GUI mainloop ended")
        except Exception as e:
            print(f"🖥️  [ERROR] GUI mainloop failed: {e}")
            import traceback
            traceback.print_exc()

class TwoPhaseBacktester:
    """
    Backtester for the two-phase trading system.
    
    Phase 1: Binary model detects moves
    Phase 2: Directional model predicts direction
    """
    
    def __init__(self, 
                 binary_model_path: str,
                 directional_model_path: str,
                 initial_capital: float = 100000.0,
                 position_size_pct: float = 0.10,
                 leverage: float = 25.0,
                 fee_rate: float = 0.0003,
                 gui=None):
        
        self.binary_model_path = binary_model_path
        self.directional_model_path = directional_model_path
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.leverage = leverage
        self.fee_rate = fee_rate
        self.gui = gui
        
        # Trading state
        self.capital = initial_capital
        self.position = None  # None, 'LONG', or 'SHORT'
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Load models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_models()
        
    def _load_models(self):
        """Load both Phase 1 and Phase 2 models"""
        print("🤖 Loading two-phase models...")
        
        # Load Phase 1 (Binary) model
        binary_checkpoint = torch.load(self.binary_model_path, map_location=self.device, weights_only=False)
        binary_args = binary_checkpoint['args']
        
        self.binary_model = BinaryMoveModel(
            input_dim=len(binary_checkpoint['feature_cols']),
            d_model=binary_args.d_model,
            num_heads=binary_args.num_heads,
            num_layers=binary_args.num_layers,
            d_ff=binary_args.d_ff,
            dropout=binary_args.dropout
        ).to(self.device)
        
        self.binary_model.load_state_dict(binary_checkpoint['model_state_dict'])
        self.binary_model.eval()
        
        # Store binary model components for feature processing
        self.binary_feature_cols = binary_checkpoint['feature_cols']
        self.binary_robust_scaler = binary_checkpoint['robust_scaler']
        self.binary_standard_scaler = binary_checkpoint['standard_scaler']
        self.binary_sequence_length = binary_args.sequence_length
        
        print(f"✅ Phase 1 (Binary) model loaded: {len(self.binary_feature_cols)} features")
        print(f"🔧 Binary sequence length from checkpoint: {self.binary_sequence_length}")
        
        # Load Phase 2 (Directional) model
        directional_checkpoint = torch.load(self.directional_model_path, map_location=self.device, weights_only=False)
        directional_args = directional_checkpoint['args']
        
        self.directional_model = TradingDirectionalModel(
            input_dim=len(directional_checkpoint['feature_cols']),
            d_model=directional_args.d_model,
            num_heads=directional_args.num_heads,
            num_layers=directional_args.num_layers,
            d_ff=directional_args.d_ff,
            dropout=directional_args.dropout
        ).to(self.device)
        
        self.directional_model.load_state_dict(directional_checkpoint['model_state_dict'])
        self.directional_model.eval()
        
        # Store directional model components
        self.directional_feature_cols = directional_checkpoint['feature_cols']
        self.directional_robust_scaler = directional_checkpoint['robust_scaler']
        self.directional_standard_scaler = directional_checkpoint['standard_scaler']
        self.directional_sequence_length = directional_args.sequence_length
        
        print(f"✅ Phase 2 (Directional) model loaded: {len(self.directional_feature_cols)} features")
        
    def predict_move(self, features: np.ndarray) -> int:
        """Phase 1: Predict if a move will happen using EXACT same method as labeling script"""
        with torch.no_grad():
            # Prepare features for binary model - pad to expected dimensions
            expected_dims = len(self.binary_feature_cols)
            if features.shape[1] != expected_dims:
                padded_features = np.zeros((features.shape[0], expected_dims))
                # Copy available features, pad the rest with zeros
                copy_dims = min(features.shape[1], expected_dims)
                padded_features[:, :copy_dims] = features[:, :copy_dims]
                features = padded_features
            
            # Create sequence
            if len(features) < self.binary_sequence_length:
                return 0  # Not enough data
                
            sequence = features[-self.binary_sequence_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Get prediction using EXACT same method as labeling script (line 100)
            outputs = self.binary_model(sequence_tensor)
            move_logits = outputs['move']
            
            # Debug: Log confidence scores
            move_probs = torch.softmax(move_logits, dim=1).cpu().numpy()[0]
            move_confidence = move_probs[1]  # Probability of move (class 1)
            
            # Use ARGMAX for predictions (same as training evaluation)
            move_prediction = np.argmax(move_logits.cpu().numpy(), axis=1)[0]
            
            # Debug: Log prediction details
            print(f"🔧 [BINARY MODEL] Logits: {move_logits.cpu().numpy()[0]}, Probs: {move_probs}, Confidence: {move_confidence:.3f}, Prediction: {move_prediction}")
            
            return move_prediction  # Returns 0 or 1 directly
    
    def predict_direction(self, features: np.ndarray) -> Tuple[str, float]:
        """Phase 2: Predict direction of detected move"""
        with torch.no_grad():
            # Prepare features for directional model - pad to expected dimensions
            expected_dims = len(self.directional_feature_cols)
            if features.shape[1] != expected_dims:
                padded_features = np.zeros((features.shape[0], expected_dims))
                # Copy available features, pad the rest with zeros
                copy_dims = min(features.shape[1], expected_dims)
                padded_features[:, :copy_dims] = features[:, :copy_dims]
                features = padded_features
            
            if len(features) < self.directional_sequence_length:
                return 'HOLD', 0.5  # Not enough data
                
            sequence = features[-self.directional_sequence_length:]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Get direction prediction
            outputs = self.directional_model(sequence_tensor)
            direction_probs = torch.softmax(outputs['direction'], dim=1)
            
            # 0=SHORT, 1=LONG
            direction_pred = torch.argmax(direction_probs, dim=1).item()
            confidence = direction_probs[0, direction_pred].item()
            
            direction = 'LONG' if direction_pred == 1 else 'SHORT'
            
            return direction, confidence
    
    def calculate_position_size(self) -> float:
        """Calculate position size based on current capital"""
        trade_capital = self.capital * self.position_size_pct
        position_value = trade_capital * self.leverage
        return position_value
    
    def calculate_fees(self, position_value: float) -> float:
        """Calculate trading fees"""
        return position_value * self.fee_rate
    
    def enter_position(self, direction: str, price: float, timestamp: pd.Timestamp, confidence: float):
        """Enter a new position"""
        if self.position is not None:
            return  # Already in position
            
        position_value = self.calculate_position_size()
        entry_fees = self.calculate_fees(position_value)
        
        # Deduct entry fees from capital
        self.capital -= entry_fees
        
        self.position = direction
        self.position_size = position_value
        self.entry_price = price
        self.entry_time = timestamp
        
        print(f"📈 ENTER {direction}: Timestamp={timestamp}, Price={price:.6f}, Size=${position_value:,.0f}, Fees=${entry_fees:.2f}, Confidence={confidence:.3f}")
    
    def exit_position(self, price: float, timestamp: pd.Timestamp, reason: str = ""):
        """Exit current position"""
        if self.position is None:
            return 0.0  # No position to exit
            
        # Calculate P&L
        if self.position == 'LONG':
            price_change = (price - self.entry_price) / self.entry_price
        else:  # SHORT
            price_change = (self.entry_price - price) / self.entry_price
            
        # Apply leverage
        pnl_before_fees = self.position_size * price_change
        
        # Calculate exit fees
        exit_fees = self.calculate_fees(self.position_size)
        
        # Net P&L after fees
        net_pnl = pnl_before_fees - exit_fees
        
        # Update capital
        self.capital += net_pnl
        
        # Record trade
        trade_duration = (timestamp - self.entry_time).total_seconds() / 60  # minutes
        
        trade_record = {
            'entry_time': self.entry_time,
            'exit_time': timestamp,
            'direction': self.position,
            'entry_price': self.entry_price,
            'exit_price': price,
            'position_size': self.position_size,
            'price_change_pct': price_change * 100,
            'pnl_before_fees': pnl_before_fees,
            'entry_fees': self.calculate_fees(self.position_size),
            'exit_fees': exit_fees,
            'net_pnl': net_pnl,
            'duration_minutes': trade_duration,
            'reason': reason
        }
        
        self.trades.append(trade_record)
        
        print(f"📉 EXIT {self.position}: Timestamp={timestamp}, Price={price:.6f}, P&L=${net_pnl:+.2f}, Duration={trade_duration:.1f}min, Reason={reason}")
        
        # Update GUI if available
        if self.gui:
            # Add trade to GUI
            self.gui.queue_update({
                'type': 'trade',
                'time': timestamp.strftime('%H:%M:%S'),
                'side': self.position,
                'entry': self.entry_price,
                'exit': price,
                'pnl': net_pnl,
                'duration': trade_duration,
                'reason': reason
            })
            
            # Update equity chart
            self.gui.queue_update({
                'type': 'equity',
                'equity': self.capital,
                'time': timestamp
            })
        
        # Reset position
        self.position = None
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        
        return net_pnl
    
    def run_backtest(self, df: pd.DataFrame, 
                    move_threshold: float = 0.7,
                    direction_threshold: float = 0.55,
                    max_hold_minutes: int = 30,
                    stop_loss_pct: float = 0.015,
                    take_profit_pct: float = 0.0035,
                    scaler_data_dir: str = None) -> Dict:
        """
        Run the two-phase backtest on the provided data.
        
        Args:
            df: DataFrame with OHLC data and features
            move_threshold: Minimum probability for Phase 1 move detection (0-1)
            direction_threshold: Minimum confidence for Phase 2 direction (0-1)
            max_hold_minutes: Maximum time to hold a position
            stop_loss_pct: Stop loss percentage (e.g., 0.015 = 1.5%)
            take_profit_pct: Take profit percentage (e.g., 0.0035 = 0.35%)
        """
        
        print(f"🚀 Running Two-Phase Backtest")
        print(f"   Period: {df['ts'].min()} to {df['ts'].max()}")
        print(f"   Samples: {len(df):,}")
        print(f"   Move threshold: {move_threshold:.1%}")
        print(f"   Direction threshold: {direction_threshold:.1%}")
        print(f"   Max hold: {max_hold_minutes} minutes")
        print(f"   Stop loss: {stop_loss_pct:.1%}")
        print(f"   Take profit: {take_profit_pct:.1%}")
        
        # Prepare features for both models
        # For binary model: use EXACT 38 features in correct order
        binary_feature_cols = [
            'open', 'volume', 'quote_volume', 'r1', 'r2', 'r5', 'r10', 
            'range_pct', 'body_pct', 'rv', 'vol_z', 'avg_trade_size',
            'buy_vol', 'sell_vol', 'mean_size', 'max_size', 'p95_size', 
            'n_trades', 'signed_vol', 'imb_aggr', 'CVD', 'signed_volatility',
            'block_trades', 'impact_proxy', 'vw_tick_return', 'vol_regime',
            'drawdown', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
            'session_asia', 'session_europe', 'session_us', 'price_position',
            'vol_concentration', 'vol_entropy', 'y_actionable'
        ]
        
        print(f"📊 Using EXACT {len(binary_feature_cols)} binary features (no truncation)")
        
        # 📊 CREATE DEBUG DATAFRAME FOR CSV EXPORT (before any processing)
        self.debug_features_df = df.copy()  # Store full dataframe with all features for export
        
        # Create binary feature matrix with exact columns
        binary_features_df = pd.DataFrame()
        missing_binary = 0
        for col in binary_feature_cols:
            if col in df.columns:
                binary_features_df[col] = df[col]
            else:
                binary_features_df[col] = 0.0
                missing_binary += 1
                print(f"⚠️  Missing binary feature: {col}")
        
        if missing_binary > 0:
            print(f"⚠️  Added {missing_binary} missing binary features as zeros")
        
        # Store for final summary
        self.missing_binary_features = missing_binary
        
        # CRITICAL FIX: Use FRESH scalers like directional labeling script
        print(f"🔧 Fitting FRESH scalers on scaler data")
        
        # Use scaler data directory if provided, otherwise use recent data
        if scaler_data_dir:
            scaler_data_path = Path(scaler_data_dir)
            clean_symbol = "doge"  # Default for DOGEUSDT
            scaler_klines_file = scaler_data_path / f"scaler_data_{clean_symbol}_klines.csv"
            scaler_agg_file = scaler_data_path / f"scaler_data_{clean_symbol}_aggTrades.csv"
            
            if scaler_klines_file.exists() and scaler_agg_file.exists():
                print(f"🔧 Using scaler data from {scaler_data_path}")
                print(f"   Klines: {scaler_klines_file}")
                print(f"   AggTrades: {scaler_agg_file}")
                
                # Load scaler data
                scaler_klines_df = pd.read_csv(scaler_klines_file)
                scaler_agg_df = pd.read_csv(scaler_agg_file)
                
                # Convert timestamps
                scaler_klines_df['ts'] = pd.to_datetime(scaler_klines_df['timestamp'], unit='ms')
                scaler_agg_df['ts'] = pd.to_datetime(scaler_agg_df['Timestamp'], unit='ms')
                
                print(f"🔧 Loaded scaler data: {len(scaler_klines_df)} klines, {len(scaler_agg_df)} aggTrades")
                print(f"📅 Scaler data range: {scaler_klines_df['ts'].min()} to {scaler_klines_df['ts'].max()}")
                
                # Use scaler data for fitting
                df_recent = scaler_klines_df.copy()
            else:
                print(f"⚠️ Scaler data files not found, using recent data instead")
                # Fall back to recent data
                latest_date = df['ts'].max()
                thirty_days_ago = latest_date - pd.DateOffset(days=30)
                df_recent = df[df['ts'] >= thirty_days_ago].copy()
                print(f"🔧 Using recent data for scaler fitting: {len(df_recent)} samples from {thirty_days_ago} to {latest_date}")
        else:
            # Use recent data for scaler fitting (last 30 days - more practical than 6 months)
            latest_date = df['ts'].max()
            thirty_days_ago = latest_date - pd.DateOffset(days=30)
            df_recent = df[df['ts'] >= thirty_days_ago].copy()
            print(f"🔧 Using recent data for scaler fitting: {len(df_recent)} samples from {thirty_days_ago} to {latest_date}")
        
        # Prepare recent data with same preprocessing
        recent_features_df = pd.DataFrame()
        for col in binary_feature_cols:
            if col in df_recent.columns:
                recent_features_df[col] = df_recent[col]
            else:
                recent_features_df[col] = 0.0
        
        # Apply same preprocessing to recent data
        recent_features_processed = recent_features_df.copy()
        
        # Define volume features for preprocessing
        volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                          'buy_vol_ratio', 'sell_vol_ratio', 'volume_ratio', 'volume_ma_ratio']
        
        for col in recent_features_processed.columns:
            if col not in volume_features and recent_features_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                p1, p99 = np.percentile(recent_features_processed[col], [1, 99])
                recent_features_processed[col] = recent_features_processed[col].clip(p1, p99)
        
        # Fit FRESH scalers on recent data
        from sklearn.preprocessing import RobustScaler, StandardScaler
        fresh_robust_scaler = RobustScaler()
        fresh_standard_scaler = StandardScaler()
        
        recent_features_robust_scaled = fresh_robust_scaler.fit_transform(recent_features_processed)
        recent_features_final = fresh_standard_scaler.fit_transform(recent_features_robust_scaled)
        
        print(f"🔧 Fresh scalers fitted on recent data")
        
        # Apply preprocessing (volume scaling, outlier clipping) exactly as in training
        binary_features_processed = binary_features_df.copy()
        # SKIP volume scaling - already done in preprocessing
        for col in binary_features_processed.columns:
            if col not in volume_features and binary_features_processed[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                p1, p99 = np.percentile(binary_features_processed[col], [1, 99])
                binary_features_processed[col] = binary_features_processed[col].clip(p1, p99)
        
        # DEBUG: Check volume feature statistics after preprocessing
        print(f"🔧 Volume features after scaling:")
        for col in ['volume', 'quote_volume', 'buy_vol', 'sell_vol']:
            if col in binary_features_processed.columns:
                mean_val = binary_features_processed[col].mean()
                std_val = binary_features_processed[col].std()
                print(f"   {col}: mean={mean_val:.6f}, std={std_val:.6f}")
        
        # Using FRESH scalers fitted on recent data (no need to fix old scalers)
        
        # DEBUG: Check data BEFORE scaling
        print(f"🔧 [BACKTESTER] Data BEFORE scaling:")
        temp_before_df = pd.DataFrame(binary_features_processed, columns=binary_feature_cols, index=binary_features_df.index)
        print(f"   volume: mean={temp_before_df['volume'].mean():.6f}, std={temp_before_df['volume'].std():.6f}")
        print(f"   impact_proxy: mean={temp_before_df['impact_proxy'].mean():.6f}, std={temp_before_df['impact_proxy'].std():.6f}")
        
        # Apply the FRESH scalers (model expects scaled input)
        features_robust_scaled = fresh_robust_scaler.transform(binary_features_processed)
        binary_features_scaled = fresh_standard_scaler.transform(features_robust_scaled)
        
        # DEBUG: Check if binary_features_scaled has variation across candles
        print(f"🔧 [DEBUG] binary_features_scaled shape: {binary_features_scaled.shape}")
        if len(binary_features_scaled) > 60:
            # Check volume variation across first 60 candles
            volume_idx = binary_feature_cols.index('volume') if 'volume' in binary_feature_cols else -1
            if volume_idx >= 0:
                first_60_volumes = binary_features_scaled[:60, volume_idx]
                print(f"🔧 [DEBUG] First 60 candles volume: min={first_60_volumes.min():.6f}, max={first_60_volumes.max():.6f}, std={first_60_volumes.std():.6f}")
                print(f"🔧 [DEBUG] First 5 volumes: {first_60_volumes[:5]}")
                print(f"🔧 [DEBUG] Last 5 volumes: {first_60_volumes[-5:]}")
        
        # DEBUG: Log actual model input values for comparison with live trader
        print(f"🔧 [BACKTESTER] ACTUAL MODEL INPUT features after prepare_features:")
        temp_scaled_df = pd.DataFrame(binary_features_scaled, columns=binary_feature_cols, index=binary_features_df.index)
        print(f"   r1: mean={temp_scaled_df['r1'].mean():.6f}, std={temp_scaled_df['r1'].std():.6f}")
        print(f"   r2: mean={temp_scaled_df['r2'].mean():.6f}, std={temp_scaled_df['r2'].std():.6f}")
        print(f"   volume: mean={temp_scaled_df['volume'].mean():.6f}, std={temp_scaled_df['volume'].std():.6f}")
        print(f"   range_pct: mean={temp_scaled_df['range_pct'].mean():.6f}, std={temp_scaled_df['range_pct'].std():.6f}")
        print(f"   impact_proxy: mean={temp_scaled_df['impact_proxy'].mean():.6f}, std={temp_scaled_df['impact_proxy'].std():.6f}")
        
        # DEBUG: Check impact_proxy distribution in backtester
        if 'impact_proxy' in binary_feature_cols:
            impact_proxy_idx = binary_feature_cols.index('impact_proxy')
            if impact_proxy_idx < binary_features_scaled.shape[1]:
                impact_proxy_values = binary_features_scaled[:, impact_proxy_idx]
                print(f"🔍 BACKTESTER impact_proxy stats (after scaling):")
                print(f"   Count: {len(impact_proxy_values)}")
                print(f"   Mean: {np.mean(impact_proxy_values):.8f}")
                print(f"   Median: {np.median(impact_proxy_values):.8f}")
                print(f"   Min: {np.min(impact_proxy_values):.8f}")
                print(f"   Max: {np.max(impact_proxy_values):.8f}")
                print(f"   Std: {np.std(impact_proxy_values):.8f}")
                print(f"   95th percentile: {np.percentile(impact_proxy_values, 95):.8f}")
                print(f"   99th percentile: {np.percentile(impact_proxy_values, 99):.8f}")
                
                # Also check raw impact_proxy before scaling
                if 'impact_proxy' in df.columns:
                    raw_impact_proxy = df['impact_proxy'].dropna()
                    print(f"🔍 BACKTESTER impact_proxy stats (before scaling):")
                    print(f"   Count: {len(raw_impact_proxy)}")
                    print(f"   Mean: {raw_impact_proxy.mean():.8f}")
                    print(f"   Median: {raw_impact_proxy.median():.8f}")
                    print(f"   Min: {raw_impact_proxy.min():.8f}")
                    print(f"   Max: {raw_impact_proxy.max():.8f}")
                    print(f"   Std: {raw_impact_proxy.std():.8f}")
        
        # Use ALL 38 features (no truncation needed - they're already exact)
        binary_features_final = binary_features_scaled
        print(f"✅ Created EXACT 38 binary features (no truncation)")
        
        # For directional model: use EXACT 238 features in correct order
        directional_feature_cols = [
            'open', 'high_x', 'low_x', 'close_x', 'volume', 'quote_volume', 'r1', 'r2', 'r5', 'r10', 'range_pct', 'body_pct', 'rv', 'vol_z', 'avg_trade_size', 'buy_vol', 'sell_vol', 'mean_size', 'max_size', 'p95_size', 'n_trades', 'signed_vol', 'imb_aggr', 'CVD', 'signed_volatility', 'block_trades', 'impact_proxy', 'vw_tick_return', 'vol_regime', 'drawdown', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos', 'session_asia', 'session_europe', 'session_us', 'price_position', 'vol_concentration', 'vol_entropy',
            'buy_sell_ratio', 'buy_sell_diff', 'buy_sell_imbalance', 'buy_momentum_5', 'sell_momentum_5', 'imbalance_momentum_5', 'imbalance_volatility_5', 'buy_momentum_10', 'sell_momentum_10', 'imbalance_momentum_10', 'imbalance_volatility_10', 'buy_momentum_20', 'sell_momentum_20', 'imbalance_momentum_20', 'imbalance_volatility_20', 'cumulative_buy_pressure', 'cumulative_sell_pressure', 'cumulative_imbalance', 'buy_acceleration', 'sell_acceleration', 'imbalance_acceleration',
            'hl_spread', 'hl_spread_norm', 'open_position', 'intrabar_momentum', 'intrabar_reversal', 'volume_price_trend', 'vpt', 'vpt_momentum', 'tick_direction', 'tick_momentum', 'tick_acceleration',
            'rsi_7', 'rsi_momentum_7', 'rsi_divergence_7', 'rsi_14', 'rsi_momentum_14', 'rsi_divergence_14', 'rsi_21', 'rsi_momentum_21', 'rsi_divergence_21', 'rsi_50', 'rsi_momentum_50', 'rsi_divergence_50',
            'macd_8_21', 'macd_signal_8_21', 'macd_histogram_8_21', 'macd_momentum_8_21', 'macd_acceleration_8_21', 'macd_12_26', 'macd_signal_12_26', 'macd_histogram_12_26', 'macd_momentum_12_26', 'macd_acceleration_12_26', 'macd_19_39', 'macd_signal_19_39', 'macd_histogram_19_39', 'macd_momentum_19_39', 'macd_acceleration_19_39',
            'bb_upper_10', 'bb_lower_10', 'bb_position_10', 'bb_width_10', 'bb_squeeze_10', 'bb_upper_20', 'bb_lower_20', 'bb_position_20', 'bb_width_20', 'bb_squeeze_20', 'bb_upper_50', 'bb_lower_50', 'bb_position_50', 'bb_width_50', 'bb_squeeze_50',
            'stoch_k_14', 'stoch_d_14', 'stoch_momentum_14', 'stoch_k_21', 'stoch_d_21', 'stoch_momentum_21', 'stoch_k_50', 'stoch_d_50', 'stoch_momentum_50',
            'volume_at_high_10', 'volume_at_low_10', 'volume_at_mid_10', 'volume_concentration_10', 'high_volume_momentum_10', 'low_volume_momentum_10', 'volume_at_high_20', 'volume_at_low_20', 'volume_at_mid_20', 'volume_concentration_20', 'high_volume_momentum_20', 'low_volume_momentum_20', 'volume_at_high_50', 'volume_at_low_50', 'volume_at_mid_50', 'volume_concentration_50', 'high_volume_momentum_50', 'low_volume_momentum_50',
            'vwap_10', 'vwap_deviation_10', 'vwap_momentum_10', 'vwap_20', 'vwap_deviation_20', 'vwap_momentum_20', 'vwap_50', 'vwap_deviation_50', 'vwap_momentum_50',
            'volume_roc_3', 'volume_acceleration_3', 'volume_roc_5', 'volume_acceleration_5', 'volume_roc_10', 'volume_acceleration_10',
            'volatility_10', 'volatility_rank_10', 'volatility_regime_10', 'volatility_20', 'volatility_rank_20', 'volatility_regime_20', 'volatility_50', 'volatility_rank_50', 'volatility_regime_50',
            'trend_strength_10', 'trend_direction_10', 'trend_momentum_10', 'trend_strength_20', 'trend_direction_20', 'trend_momentum_20', 'trend_strength_50', 'trend_direction_50', 'trend_momentum_50',
            'mean_reversion_5', 'mean_reversion_momentum_5', 'zscore_5', 'mean_reversion_10', 'mean_reversion_momentum_10', 'zscore_10', 'mean_reversion_20', 'mean_reversion_momentum_20', 'zscore_20',
            'volume_efficiency_5', 'volume_efficiency_momentum_5', 'volume_efficiency_10', 'volume_efficiency_momentum_10', 'volume_efficiency_20', 'volume_efficiency_momentum_20',
            'volume_zscore', 'large_trade', 'large_buy_trade', 'large_sell_trade', 'cumulative_large_buys', 'cumulative_large_sells', 'large_trade_imbalance',
            'volume_impact_1', 'impact_decay_1', 'volume_impact_3', 'impact_decay_3', 'volume_impact_5', 'impact_decay_5',
            'price_volume_divergence_5', 'price_volume_divergence_10', 'price_volume_divergence_20', 'macd_divergence',
            'price_autocorr_1', 'volume_autocorr_1', 'price_autocorr_2', 'volume_autocorr_2', 'price_autocorr_3', 'volume_autocorr_3', 'price_autocorr_5', 'volume_autocorr_5',
            'price_volume_corr_10', 'price_volume_corr_momentum_10', 'price_volume_corr_20', 'price_volume_corr_momentum_20', 'price_volume_corr_50', 'price_volume_corr_momentum_50',
            'momentum_corr_5_10', 'momentum_corr_10_20', 'momentum_corr_20_50',
            'hour', 'minute', 'day_of_week', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'asian_session', 'european_session', 'us_session', 'session_open', 'session_close',
            'high_y', 'low_y', 'close_y'
        ]
        
        print(f"📊 Using EXACT {len(directional_feature_cols)} directional features (no truncation)")
        
        # Create directional feature matrix with exact columns
        directional_features_ordered = pd.DataFrame(index=df.index)
        missing_count = 0
        for col in directional_feature_cols:
            if col in df.columns:
                directional_features_ordered[col] = df[col]
            else:
                directional_features_ordered[col] = 0.0
                missing_count += 1
                print(f"⚠️  Missing directional feature: {col}")
                
        if missing_count > 0:
            print(f"⚠️  Added {missing_count} missing directional features as zeros")
        
        # Store for final summary
        self.missing_directional_features = missing_count
        
        # Apply same preprocessing as training
        volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                          'max_size', 'p95_size', 'signed_vol', 'dCVD', 'CVD']
        
        # SKIP volume scaling - already done in preprocessing
        for col in directional_features_ordered.columns:
            if col not in volume_features and directional_features_ordered[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                p1, p99 = np.percentile(directional_features_ordered[col], [1, 99])
                directional_features_ordered[col] = directional_features_ordered[col].clip(p1, p99)
        
        # Fill NaN values
        directional_features_ordered = directional_features_ordered.fillna(method='ffill').fillna(0)
        
        # CRITICAL FIX: Handle near-zero scale values in RobustScaler
        # The RobustScaler was fitted on training data with some features having zero variance,
        # causing scale values near zero (e.g., 4e-10). This collapses live data to constants.
        
        # Create a copy of the scaler and fix near-zero scales
        fixed_directional_scaler = self.directional_robust_scaler
        if hasattr(fixed_directional_scaler, 'scale_'):
            # Replace scales that are too small with a reasonable default
            min_scale = 1e-6  # Minimum scale threshold
            fixed_directional_scaler.scale_[fixed_directional_scaler.scale_ < min_scale] = min_scale
        
        # Apply the FIXED scalers (model expects scaled input)
        features_robust_scaled = fixed_directional_scaler.transform(directional_features_ordered)
        directional_features_final = self.directional_standard_scaler.transform(features_robust_scaled)
        
        print(f"✅ Features prepared for both phases")
        
        # Mirror live: mark candles that have aggtrade activity
        df['has_trades'] = ((df['buy_vol'] != 0) | (df['sell_vol'] != 0)).astype(int)
        
        # Main backtesting loop
        total_bars = len(df)
        start_bar = max(self.binary_sequence_length, self.directional_sequence_length)
        
        for i in range(start_bar, total_bars):
            current_row = df.iloc[i]
            current_price = current_row['close']
            current_time = current_row['ts']
            
            # Skip if current candle has no trade data (same as live trader)
            if (current_row['buy_vol'] == 0) and (current_row['sell_vol'] == 0):
                continue
            
            # Update GUI progress every 1000 bars
            if self.gui and i % 1000 == 0:
                progress = ((i - start_bar) / (total_bars - start_bar)) * 100
                winning_trades = sum(1 for t in self.trades if t['net_pnl'] > 0)
                win_rate = (winning_trades / len(self.trades) * 100) if self.trades else 0
                
                self.gui.queue_update({
                    'type': 'progress',
                    'progress': progress,
                    'status': f"Processing bar {i:,}/{total_bars:,} - {current_time.strftime('%Y-%m-%d %H:%M')}"
                })
                
                self.gui.queue_update({
                    'type': 'stats',
                    'equity': self.capital,
                    'pnl': self.capital - self.initial_capital,
                    'fees': sum(t['entry_fees'] + t['exit_fees'] for t in self.trades),
                    'trades': len(self.trades),
                    'win_rate': win_rate,
                    'position': self.position or 'NONE'
                })
            
            # Update equity curve
            if self.position is not None:
                # Calculate unrealized P&L
                if self.position == 'LONG':
                    unrealized_pnl = self.position_size * ((current_price - self.entry_price) / self.entry_price)
                else:  # SHORT
                    unrealized_pnl = self.position_size * ((self.entry_price - current_price) / self.entry_price)
                
                current_equity = self.capital + unrealized_pnl
            else:
                current_equity = self.capital
                
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': current_equity,
                'position': self.position,
                'price': current_price
            })
            
            # Check exit conditions if in position
            if self.position is not None:
                # Stop loss / Take profit using highs and lows (realistic)
                current_high = current_row['high']
                current_low = current_row['low']
                
                if self.position == 'LONG':
                    # Only check take profit with current bar's high (NO STOP LOSS)
                    target_price = self.entry_price * (1 + take_profit_pct)
                    if current_high >= target_price:
                        self.exit_position(target_price, current_time, "TAKE_PROFIT")
                        continue
                else:  # SHORT
                    # Only check take profit with current bar's low (NO STOP LOSS)
                    target_price = self.entry_price * (1 - take_profit_pct)
                    if current_low <= target_price:
                        self.exit_position(target_price, current_time, "TAKE_PROFIT")
                        continue

                # Time-based exit (evaluate after TP/SL)
                time_in_position = (current_time - self.entry_time).total_seconds() / 60
                if time_in_position >= max_hold_minutes:
                    self.exit_position(current_price, current_time, "TIME_LIMIT")
                    continue
            
            # Only look for new entries if not in position
            if self.position is None:
                # Require last N minutes to have aggtrade activity (like live but more lenient for backtesting):
                # Count backward from current bar; break immediately on first gap
                consecutive_with_trades = 0
                for k in range(self.binary_sequence_length):
                    row = df.iloc[i - k]
                    if (row['buy_vol'] != 0) or (row['sell_vol'] != 0):
                        consecutive_with_trades += 1
                    else:
                        break
                
                # Debug logging for first few attempts
                if i < 10:  # Only log first 10 attempts to avoid spam
                    print(f"🔧 Candle {i} ({current_time}): consecutive_with_trades={consecutive_with_trades}/{self.binary_sequence_length}")
                
                # Much more lenient: require at least 10 out of 60 candles to have trades (17% threshold)
                # This matches live trader behavior better for historical data with sparse trade activity
                min_required = max(10, int(self.binary_sequence_length * 0.17))
                if consecutive_with_trades < min_required:
                    continue
                # Phase 1: Check if move is detected using EXACT same method as labeling script
                binary_sequence = binary_features_final[i-self.binary_sequence_length+1:i+1]
                
                # DEBUG: Check what the model is predicting
                if i < 100:  # Only log first 100 attempts to avoid spam
                    with torch.no_grad():
                        binary_input = torch.FloatTensor(binary_sequence).unsqueeze(0).to(self.device)
                        outputs = self.binary_model(binary_input)
                        move_logits = outputs['move']
                        
                        # Get probability using softmax (same as predict_move method)
                        move_probs = torch.softmax(move_logits, dim=1).cpu().numpy()[0]
                        move_confidence = move_probs[1]  # Probability of move (class 1)
                        
                        # Use ARGMAX for prediction (same as predict_move method)
                        move_prediction = np.argmax(move_logits.cpu().numpy(), axis=1)[0]
                        
                        # Debug: Log prediction details for first few attempts
                        if i < 10:  # Only log first 10 attempts to avoid spam
                            print(f"🔧 Binary logits: {move_logits.cpu().numpy()[0]}")
                            print(f"🔧 Binary probs: {move_probs}")
                            print(f"🔧 Binary confidence: {move_confidence:.6f} (threshold: {move_threshold:.2f})")
                            print(f"🔧 Binary prediction: {move_prediction}")
                            
                            if move_prediction == 1:
                                print(f"🔧 ✅ MOVE DETECTED! Prediction: {move_prediction}, Confidence: {move_confidence:.6f}")
                            else:
                                print(f"🔧 ❌ NO MOVE - Prediction: {move_prediction}, Confidence: {move_confidence:.6f}")
                
        # DEBUG: Check sequence shape and content
        if i < 100:  # Only log first 100 attempts to avoid spam
            print(f"🔧 [DEBUG] Binary sequence shape: {binary_sequence.shape}")
            print(f"🔧 [DEBUG] Binary sequence range: [{i-self.binary_sequence_length+1}:{i+1}]")
            print(f"🔧 [DEBUG] Binary sequence sample (first candle): {binary_sequence[0][:5]}")  # First 5 features
            print(f"🔧 [DEBUG] Binary sequence sample (last candle): {binary_sequence[-1][:5]}")  # First 5 features
            
            # Check if volume features are identical across sequence
            volume_idx = self.binary_feature_cols.index('volume') if 'volume' in self.binary_feature_cols else -1
            if volume_idx >= 0:
                volume_values = binary_sequence[:, volume_idx]
                print(f"🔧 [DEBUG] Volume values across sequence: min={volume_values.min():.6f}, max={volume_values.max():.6f}, std={volume_values.std():.6f}")
                print(f"🔧 [DEBUG] Volume values first 5: {volume_values[:5]}")
                print(f"🔧 [DEBUG] Volume values last 5: {volume_values[-5:]}")
                
                # DEBUG: Check if we're extracting the same row multiple times
                print(f"🔧 [DEBUG] Checking if sequence extraction is correct:")
                print(f"🔧 [DEBUG] binary_features_final shape: {binary_features_final.shape}")
                print(f"🔧 [DEBUG] Sample from binary_features_final[{i-59}:{i+1}] volume values:")
                sample_volumes = binary_features_final[i-59:i+1, volume_idx]
                print(f"🔧 [DEBUG] Sample volumes: min={sample_volumes.min():.6f}, max={sample_volumes.max():.6f}, std={sample_volumes.std():.6f}")
                print(f"🔧 [DEBUG] Sample volumes first 5: {sample_volumes[:5]}")
                print(f"🔧 [DEBUG] Sample volumes last 5: {sample_volumes[-5:]}")
                
                # DEBUG: Check raw data before any processing
                print(f"🔧 [DEBUG] Checking raw data before processing:")
                raw_volume_idx = self.binary_feature_cols.index('volume') if 'volume' in self.binary_feature_cols else -1
                if raw_volume_idx >= 0:
                    raw_sample_volumes = binary_features_processed[i-59:i+1, raw_volume_idx]
                    print(f"🔧 [DEBUG] Raw sample volumes: min={raw_sample_volumes.min():.6f}, max={raw_sample_volumes.max():.6f}, std={raw_sample_volumes.std():.6f}")
                    print(f"🔧 [DEBUG] Raw sample volumes first 5: {raw_sample_volumes[:5]}")
                    print(f"🔧 [DEBUG] Raw sample volumes last 5: {raw_sample_volumes[-5:]}")

                # Export exact binary inputs (flattened) sent to model
                try:
                    export_row = {"candle_time": str(current_time), "type": "binary_input"}
                    # Flatten with featureName_t-59 ... featureName_t
                    for t in range(-self.binary_sequence_length, 0):
                        for j, col in enumerate(self.binary_feature_cols):
                            export_row[f"{col}_{t}"] = float(binary_sequence[t][j])
                    out_path = os.path.join("Live", "DOGE", "backtest_binary_inputs.csv")
                    pd.DataFrame([export_row]).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
                except Exception as e:
                    print(f"[CSV] Failed to write backtest binary_input: {e}")

                move_prediction = self.predict_move(binary_sequence)
                
                # Debug: Log prediction details for first few attempts
                if i < 100:  # Only log first 100 attempts to avoid spam
                    print(f"🔧 [DEBUG] Candle {i} ({current_time}): Binary prediction = {move_prediction}")
                
                if move_prediction == 1:  # Move detected (same as labeling script)
                    # Phase 2: Get direction prediction using ARGMAX (no threshold!)
                    directional_sequence = directional_features_final[i-self.directional_sequence_length+1:i+1]

                    # Export exact directional inputs (flattened)
                    try:
                        export_row = {"candle_time": str(current_time), "type": "directional_input"}
                        for k, val in enumerate(directional_sequence.reshape(-1).tolist()):
                            export_row[f"f{k}"] = float(val)
                        out_path = os.path.join("Live", "DOGE", "backtest_directional_inputs.csv")
                        pd.DataFrame([export_row]).to_csv(out_path, mode='a', header=not os.path.exists(out_path), index=False)
                    except Exception as e:
                        print(f"[CSV] Failed to write backtest directional_input: {e}")

                    direction, direction_pred = self.predict_direction(directional_sequence)
                    
                    if direction != 'HOLD':  # Always take the ARGMAX prediction
                        # Enter position with ARGMAX confidence
                        self.enter_position(direction, current_price, current_time, direction_pred)
        
        # Close any remaining position
        if self.position is not None:
            final_price = df.iloc[-1]['close']
            final_time = df.iloc[-1]['ts']
            self.exit_position(final_price, final_time, "END_OF_DATA")
        
        return self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'final_capital': self.capital,
                'error': 'No trades executed'
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = (trades_df['net_pnl'] > 0).sum()
        losing_trades = (trades_df['net_pnl'] <= 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['net_pnl'].sum()
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Return metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        # Trade distribution
        long_trades = (trades_df['direction'] == 'LONG').sum()
        short_trades = (trades_df['direction'] == 'SHORT').sum()
        
        # Duration metrics
        avg_duration = trades_df['duration_minutes'].mean()
        max_duration = trades_df['duration_minutes'].max()
        min_duration = trades_df['duration_minutes'].min()
        
        # Drawdown calculation
        equity_df = pd.DataFrame(self.equity_curve)
        if len(equity_df) > 0:
            equity_df['running_max'] = equity_df['equity'].expanding().max()
            equity_df['drawdown'] = (equity_df['equity'] - equity_df['running_max']) / equity_df['running_max']
            max_drawdown = equity_df['drawdown'].min()
        else:
            max_drawdown = 0.0
        
        # Exit reason analysis
        exit_reasons = trades_df['reason'].value_counts().to_dict()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_capital': self.capital,
            'initial_capital': self.initial_capital,
            'avg_duration_minutes': avg_duration,
            'max_duration_minutes': max_duration,
            'min_duration_minutes': min_duration,
            'max_drawdown': max_drawdown,
            'exit_reasons': exit_reasons,
            'trades': trades_df.to_dict('records')
        }

def load_live_data(use_binance=False) -> pd.DataFrame:
    """Load live data from CSV files in Live/DOGE/ directory"""
    import os
    
    if use_binance:
        print(f"🔴 Loading BINANCE LIVE data from CSV files...")
        # Check if we're running from Live/DOGE directory or root directory
        if os.path.exists("binancedatadoge.csv"):
            ohlcv_path = "binancedatadoge.csv"
            trades_path = "binanceaggtradesdoge.csv"
        else:
            ohlcv_path = "Live/DOGE/binancedatadoge.csv"
            trades_path = "Live/DOGE/binanceaggtradesdoge.csv"
    else:
        print(f"🔴 Loading MEXC LIVE data from CSV files...")
        # Check if we're running from Live/DOGE directory or root directory
        if os.path.exists("datadoge.csv"):
            ohlcv_path = "datadoge.csv"
            trades_path = "aggtradesdoge.csv"
        else:
            ohlcv_path = "Live/DOGE/datadoge.csv"
            trades_path = "Live/DOGE/aggtradesdoge.csv"
    
    # Load OHLCV data
    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"Live OHLCV data not found: {ohlcv_path}")
    
    ohlcv_df = pd.read_csv(ohlcv_path)
    print(f"✅ Loaded OHLCV data: {len(ohlcv_df)} candles")
    
    # Load aggregate trades data
    if not os.path.exists(trades_path):
        raise FileNotFoundError(f"Live trades data not found: {trades_path}")
    
    trades_df = pd.read_csv(trades_path)
    print(f"✅ Loaded trades data: {len(trades_df)} trades")
    
    # Convert column names to lowercase to match expected format
    ohlcv_df.columns = ohlcv_df.columns.str.lower()
    trades_df.columns = trades_df.columns.str.lower()
    
    # Ensure timestamp columns are datetime
    # OHLCV timestamps are in milliseconds since epoch
    ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
    # Use the 'datetime' column for trades as it's already in proper format
    trades_df['timestamp'] = pd.to_datetime(trades_df['datetime'])
    
    # Process features similar to live script
    print(f"🔧 Processing features from live data...")
    
    # Create basic features first
    df = ohlcv_df.copy()
    df['ts'] = df['timestamp']
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float) 
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # CRITICAL FIX: Scale live data to match RobustScaler expectations
    # The RobustScaler was fitted on data with volume center=2.35, but live data has different scale
    if not use_binance:
        # MEXC data: scale to match RobustScaler expectations
        df['volume'] = df['volume'] * 599.88
    else:
        # Binance data: use raw volume data
        pass
    
    # Add basic technical indicators
    df['r1'] = df['close'].pct_change()
    df['r5'] = df['close'].pct_change(5)
    df['r15'] = df['close'].pct_change(15)
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Additional price features
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['co_ratio'] = (df['close'] - df['open']) / df['open']
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['price_ma'] = df['close'].rolling(20).mean()
    df['volatility'] = df['r1'].rolling(20).std()
    
    # Quote volume (approximate)
    df['quote_volume'] = df['volume'] * df['close']
    
    # Calculate trade-based features
    df = calculate_trade_features_for_backtest(df, trades_df)
    
    # Debug: Check imb_aggr immediately after trade processing
    print(f"🔧 Debug: imb_aggr immediately after trade processing:")
    print(f"   Count: {len(df)}, Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_data_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0) | (df['signed_vol'] != 0)
    print(f"   Trade data candles: {trade_data_mask.sum()}")
    if trade_data_mask.sum() > 0:
        sample_with_trades = df[trade_data_mask]['imb_aggr'].head()
        print(f"   Sample values (candles with trades): {sample_with_trades.tolist()}")
    
    # Add missing columns with defaults
    if 'y_actionable' not in df.columns:
        df['y_actionable'] = 0
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    # ==== HARDCODE ALL 38 BINARY FEATURES EXACTLY ====
    # Binary model expects these EXACT 38 features:
    # 0: open, 1: volume, 2: quote_volume, 3: r1, 4: r2, 5: r5, 6: r10, 
    # 7: range_pct, 8: body_pct, 9: rv, 10: vol_z, 11: avg_trade_size,
    # 12: buy_vol, 13: sell_vol, 14: mean_size, 15: max_size, 16: p95_size, 
    # 17: n_trades, 18: signed_vol, 19: imb_aggr, 20: CVD, 21: signed_volatility,
    # 22: block_trades, 23: impact_proxy, 24: vw_tick_return, 25: vol_regime,
    # 26: drawdown, 27: minute_sin, 28: minute_cos, 29: day_sin, 30: day_cos,
    # 31: session_asia, 32: session_europe, 33: session_us, 34: price_position,
    # 35: vol_concentration, 36: vol_entropy, 37: y_actionable
    
    print("🔧 Computing ALL 38 binary features exactly...")
    
    # Basic returns (already have r1)
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
    minute_of_day = df['ts'].dt.hour * 60 + df['ts'].dt.minute
    day_of_week = df['ts'].dt.dayofweek
    df['minute_sin'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df['minute_cos'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)

    # Sessions
    hour = df['ts'].dt.hour
    df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['session_us'] = ((hour >= 16) & (hour < 24)).astype(int)

    # Price position and volume features
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['vol_concentration'] = df['volume'] / (df['volume'].rolling(20, min_periods=5).mean() + 1e-9)
    df['vol_entropy'] = -(df['vol_concentration']) * np.log(df['vol_concentration'] + 1e-12)
    
    # y_actionable = 0 (will be added later)
    df['y_actionable'] = 0

    # Debug: Check imb_aggr before directional features
    print(f"🔧 Debug: imb_aggr before directional features calculation:")
    print(f"   Count: {len(df)}, Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_data_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0) | (df['signed_vol'] != 0)
    if trade_data_mask.sum() > 0:
        sample_with_trades = df[trade_data_mask]['imb_aggr'].head()
        print(f"   Sample values (candles with trades): {sample_with_trades.tolist()}")
    
    # === NOW COMPUTE ALL 238 DIRECTIONAL FEATURES ===
    df = compute_all_directional_features(df)
    
    # Debug: Check imb_aggr before NaN/inf replacement
    print(f"🔧 Debug: imb_aggr before NaN/inf replacement:")
    print(f"   Count: {len(df)}, Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0)
    if trade_mask.sum() > 0:
        print(f"   Sample values (candles with trades): {df[trade_mask]['imb_aggr'].head().tolist()}")
    else:
        print(f"   Sample values (first 5): {df['imb_aggr'].head().tolist()}")
    print(f"   NaN count: {df['imb_aggr'].isna().sum()}")
    print(f"   Inf count: {np.isinf(df['imb_aggr']).sum()}")
    
    # Fill NaNs and infinities
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Debug: Check imb_aggr after NaN/inf replacement
    print(f"🔧 Debug: imb_aggr after NaN/inf replacement:")
    print(f"   Count: {len(df)}, Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0)
    if trade_mask.sum() > 0:
        print(f"   Sample values (candles with trades): {df[trade_mask]['imb_aggr'].head().tolist()}")
    else:
        print(f"   Sample values (first 5): {df['imb_aggr'].head().tolist()}")

    # Apply same preprocessing as live script
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'mean_size', 'signed_vol', 'dCVD', 'CVD', 'signed_volatility']
    
    # EXPERIMENT: Skip scaling factor since we're testing without scalers
    # if use_binance:
    #     print(f"🔧 Applying Binance scaling factor (1.67) to volume features...")
    #     for col in df.columns:
    #         if col in volume_features:
    #             df[col] = df[col] * 1.67
    #             print(f"   Applied 1.67 scaling to {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")
    
    # DEBUG: Check raw volume values BEFORE any scaling
    print(f"🔧 [DEBUG] RAW volume values BEFORE any scaling:")
    for col in ['volume', 'buy_vol', 'sell_vol', 'tot_vol']:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    print(f"🔧 Applying preprocessing (SKIPPING volume scaling - already in correct units)...")
    for col in df.columns:
        if col in volume_features:
            print(f"   {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f} (no scaling applied)")
        elif col == 'impact_proxy':
            # Set impact_proxy to 0 to match training data (as done in live script)
            df[col] = 0.0
            print(f"   Set {col} to 0 (matching training data)")
        elif df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Apply outlier clipping
            p1, p99 = np.percentile(df[col].dropna(), [1, 99])
            df[col] = df[col].clip(p1, p99)
    
    # DO NOT filter out candles. Keep full minute bars for realistic exits.
    print(f"✅ Live data processed: {len(df)} samples with {len(df.columns)} features (no candle filtering)")
    print(f"   Period: {df['ts'].min()} to {df['ts'].max()}")
    print(f"   Feature ranges after preprocessing:")
    for col in ['r1', 'volume', 'buy_vol', 'sell_vol', 'impact_proxy', 'signed_volatility']:
        if col in df.columns:
            print(f"     {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")
    
    return df


def compute_all_directional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ALL 238 directional features exactly as expected by the model"""
    
    print(f"🔧 Computing ALL 238 directional features...")
    
    # Debug: Check imb_aggr at start of function
    print(f"🔧 [DIRECTIONAL] imb_aggr at START: Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0)
    if trade_mask.sum() > 0:
        print(f"   Sample values: {df[trade_mask]['imb_aggr'].head().tolist()}")
    
    # === DIRECTIONAL MODEL SPECIFIC FEATURES ===
    # Directional model has high_x, low_x, close_x (vs high, low, close for binary)
    df['high_x'] = df['high']
    df['low_x'] = df['low'] 
    df['close_x'] = df['close']
    
    # Buy/sell ratios and imbalances
    df['buy_sell_ratio'] = df['buy_vol'] / (df['sell_vol'] + 1e-9)
    df['buy_sell_diff'] = df['buy_vol'] - df['sell_vol']
    # Use same formula as imb_aggr to avoid floating-point differences
    df['buy_sell_imbalance'] = df.apply(
        lambda row: (row['buy_vol'] - row['sell_vol']) / (row['buy_vol'] + row['sell_vol']) 
        if (row['buy_vol'] + row['sell_vol']) > 0 else 0.0, axis=1
    )
    
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
    df['hour'] = df['ts'].dt.hour
    df['minute'] = df['ts'].dt.minute
    df['day_of_week'] = df['ts'].dt.dayofweek
    
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
    
    print(f"✅ Computed ALL directional features")
    
    # Debug: Check imb_aggr at END of function
    print(f"🔧 [DIRECTIONAL] imb_aggr at END: Non-zero: {(df['imb_aggr'] != 0).sum()}")
    trade_mask = (df['buy_vol'] != 0) | (df['sell_vol'] != 0)
    if trade_mask.sum() > 0:
        print(f"   Sample values: {df[trade_mask]['imb_aggr'].head().tolist()}")
    
    return df


def calculate_trade_features_for_backtest(ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trade-based features for each candle using aggregate trades data"""
    
    print(f"🔧 Calculating trade features from {len(trades_df)} trades...")
    
    # Group trades by minute candles
    # Binance OHLCV timestamps end at :59 seconds, so we need to match trades within that minute
    # For example, candle ending at 17:07:59 should match trades from 17:07:00 to 17:07:59
    trades_df['candle_time'] = trades_df['timestamp'].dt.floor('1min')
    
    print(f"🔧 Debug: Trade time range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
    print(f"🔧 Debug: OHLCV time range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
    print(f"🔧 Debug: Unique candle times in trades: {len(trades_df['candle_time'].unique())}")
    print(f"🔧 Debug: Sample trade candle times: {trades_df['candle_time'].head(3).tolist()}")
    print(f"🔧 Debug: Sample OHLCV times: {ohlcv_df['timestamp'].head(3).tolist()}")
    
    # Initialize trade feature columns (EXCEPT cumulative ones like CVD)
    trade_features = {
        'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
        'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
        'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
        'signed_volatility': 0.0, 'impact_proxy': 0.0,
        'vw_tick_return': 0.0, 'rv': 0.0
    }
    
    # Initialize non-cumulative features only
    for col in trade_features:
        ohlcv_df[col] = trade_features[col]
    
    # Initialize cumulative features properly (don't overwrite existing values)
    if 'CVD' not in ohlcv_df.columns:
        ohlcv_df['CVD'] = 0.0
    if 'dCVD' not in ohlcv_df.columns:
        ohlcv_df['dCVD'] = 0.0
    
    # Process each candle
    cvd_cumulative = 0.0
    prev_cvd = 0.0
    candles_processed = 0
    
    for idx, candle in ohlcv_df.iterrows():
        # For each candle ending at candle['timestamp'], find trades within that minute
        # Binance OHLCV timestamps end at :59 seconds
        candle_end = candle['timestamp']
        candle_start = candle_end.replace(second=0, microsecond=0)
        
        # Match trades that fall within this candle's time window
        candle_trades = trades_df[
            (trades_df['timestamp'] >= candle_start) & 
            (trades_df['timestamp'] <= candle_end)
        ]
        
        if len(candle_trades) > 0:
            candles_processed += 1
            # Calculate volumes
            buy_trades = candle_trades[candle_trades['isbuyermaker'] == False]  # Taker buy = aggressive buy
            sell_trades = candle_trades[candle_trades['isbuyermaker'] == True]   # Taker sell = aggressive sell
            
            buy_vol = buy_trades['quantity'].sum() if len(buy_trades) > 0 else 0.0
            sell_vol = sell_trades['quantity'].sum() if len(sell_trades) > 0 else 0.0
            tot_vol = candle_trades['quantity'].sum()
            
            # Size statistics
            sizes = candle_trades['quantity']
            mean_size = sizes.mean() if len(sizes) > 0 else 0.0
            max_size = sizes.max() if len(sizes) > 0 else 0.0
            p95_size = sizes.quantile(0.95) if len(sizes) > 0 else 0.0
            n_trades = len(candle_trades)
            
            # Signed volume and imbalance
            signed_vol = buy_vol - sell_vol
            imb_aggr = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0.0
            
            # Debug first few candles
            if candles_processed <= 5:
                print(f"🔧 Debug candle {candles_processed} (idx={idx}): {candle_end} - {len(candle_trades)} trades, buy_vol={buy_vol:.6f}, sell_vol={sell_vol:.6f}, imb_aggr={imb_aggr:.6f}")
            
            # Block trades (large trades > 95th percentile)
            if p95_size > 0:
                block_trades = len(sizes[sizes >= p95_size])
            else:
                block_trades = 0
            
            # CVD (Cumulative Volume Delta)
            cvd_cumulative += signed_vol
            dCVD = cvd_cumulative - prev_cvd
            prev_cvd = cvd_cumulative
            
            # Signed volatility (volatility weighted by order flow)
            price_changes = candle_trades['price'].diff().abs()
            if len(price_changes) > 1 and signed_vol != 0:
                signed_volatility = price_changes.mean() * np.sign(signed_vol)
            else:
                signed_volatility = 0.0
            
            # Volume-weighted tick return (per bar)
            tick_returns = candle_trades['price'].pct_change().fillna(0.0)
            qty = candle_trades['quantity'].fillna(0.0)
            vw_tick_return = (tick_returns * qty).sum() / (qty.sum() + 1e-9)

            # Realized volatility (intra-trade) within the bar
            rv = (tick_returns ** 2).sum()

            # Impact proxy (similar to live script calculation)
            r1 = candle['close'] / candle['open'] - 1 if candle['open'] > 0 else 0
            impact_proxy = abs(r1) / (tot_vol + 1e-9) if tot_vol > 0 else 0.0
            
            # Update dataframe
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
    
    print(f"✅ Trade features calculated for {len(ohlcv_df)} candles ({candles_processed} with actual trade data)")
    return ohlcv_df


def load_raw_data(symbol: str = "DOGEUSDT", interval: str = "1m") -> pd.DataFrame:
    """Load raw OHLCV and aggtrades from data_parquet/ and compute features internally"""
    import os
    
    print(f"🔴 Loading RAW data from parquet files...")
    
    # Load raw OHLCV data
    ohlcv_path = f"data_parquet/klines_{symbol}_{interval}.parquet"
    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"Raw OHLCV data not found: {ohlcv_path}")
    
    ohlcv_df = pd.read_parquet(ohlcv_path)
    print(f"✅ Loaded raw OHLCV data: {len(ohlcv_df)} candles")
    
    # Load raw aggregate trades data
    trades_path = f"data_parquet/agg_{symbol}.parquet"
    if not os.path.exists(trades_path):
        raise FileNotFoundError(f"Raw trades data not found: {trades_path}")
    
    trades_df = pd.read_parquet(trades_path)
    print(f"✅ Loaded raw trades data: {len(trades_df)} trades")
    
    # Convert timestamps to datetime and rename columns
    # OHLCV uses 'open_time' 
    ohlcv_df['ts'] = pd.to_datetime(ohlcv_df['open_time'])
    
    # Trades uses 'transact_time'
    trades_df['timestamp'] = pd.to_datetime(trades_df['transact_time'])
    
    # Filter to 2025 data only
    ohlcv_2025 = ohlcv_df[ohlcv_df['ts'].dt.year == 2025].copy()
    trades_2025 = trades_df[trades_df['timestamp'].dt.year == 2025].copy()
    
    print(f"✅ Filtered to 2025: {len(ohlcv_2025)} candles, {len(trades_2025)} trades")
    
    # Process features from raw data (same as live data processing)
    print(f"🔧 Processing features from raw parquet data...")
    
    # Create basic features first
    df = ohlcv_2025.copy()
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float) 
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # Add basic technical indicators
    df['r1'] = df['close'].pct_change()
    df['r5'] = df['close'].pct_change(5)
    df['r15'] = df['close'].pct_change(15)
    
    # RSI calculation
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Additional price features
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['co_ratio'] = (df['close'] - df['open']) / df['open']
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['price_ma'] = df['close'].rolling(20).mean()
    df['volatility'] = df['r1'].rolling(20).std()
    
    # Quote volume (approximate)
    df['quote_volume'] = df['volume'] * df['close']
    
    # Calculate trade-based features using raw trades
    df = calculate_trade_features_from_raw_parquet(df, trades_2025)
    
    # ==== HARDCODE ALL 38 BINARY FEATURES EXACTLY ====
    print("🔧 Computing ALL 38 binary features exactly...")
    
    # Basic returns (already have r1)
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
    minute_of_day = df['ts'].dt.hour * 60 + df['ts'].dt.minute
    day_of_week = df['ts'].dt.dayofweek
    df['minute_sin'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
    df['minute_cos'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))
    df['day_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Sessions
    hour = df['ts'].dt.hour
    df['session_asia'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['session_europe'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['session_us'] = ((hour >= 16) & (hour < 24)).astype(int)
    
    # Price position and volume features
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)
    df['vol_concentration'] = df['volume'] / (df['volume'].rolling(20, min_periods=5).mean() + 1e-9)
    df['vol_entropy'] = -(df['vol_concentration']) * np.log(df['vol_concentration'] + 1e-12)
    
    # y_actionable = 0 (will be added later)
    df['y_actionable'] = 0
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    # Compute ALL directional features
    df = compute_all_directional_features(df)
    
    # Fill NaNs and infinities
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Apply same preprocessing as live script
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'mean_size', 'signed_vol', 'dCVD', 'CVD', 'signed_volatility']
    
    # EXPERIMENT: Skip scaling factor since we're testing without scalers
    # if use_binance:
    #     print(f"🔧 Applying Binance scaling factor (1.67) to volume features...")
    #     for col in df.columns:
    #         if col in volume_features:
    #             df[col] = df[col] * 1.67
    #             print(f"   Applied 1.67 scaling to {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")
    
    # DEBUG: Check raw volume values BEFORE any scaling
    print(f"🔧 [DEBUG] RAW volume values BEFORE any scaling:")
    for col in ['volume', 'buy_vol', 'sell_vol', 'tot_vol']:
        if col in df.columns:
            print(f"   {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}")
    
    print(f"🔧 Applying preprocessing (SKIPPING volume scaling - already in correct units)...")
    for col in df.columns:
        if col in volume_features:
            print(f"   {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f} (no scaling applied)")
        elif col == 'impact_proxy':
            # Set impact_proxy to 0 to match training data (as done in live script)
            df[col] = 0.0
            print(f"   Set {col} to 0 (matching training data)")
        elif df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Apply outlier clipping
            p1, p99 = np.percentile(df[col].dropna(), [1, 99])
            df[col] = df[col].clip(p1, p99)
    
    print(f"✅ Raw data processed: {len(df)} samples with {len(df.columns)} features")
    print(f"   Period: {df['ts'].min()} to {df['ts'].max()}")
    
    return df


def calculate_trade_features_from_raw_parquet(ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trade-based features from raw parquet trades data"""
    
    print(f"🔧 Calculating trade features from {len(trades_df)} raw trades...")
    
    # Group trades by minute candles
    trades_df['candle_time'] = trades_df['timestamp'].dt.floor('1min')
    
    # Initialize trade feature columns (EXCEPT cumulative ones like CVD)
    trade_features = {
        'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
        'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
        'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
        'signed_volatility': 0.0, 'impact_proxy': 0.0,
        'vw_tick_return': 0.0, 'rv': 0.0
    }
    
    # Initialize non-cumulative features only
    for col in trade_features:
        ohlcv_df[col] = trade_features[col]
    
    # Initialize cumulative features properly (don't overwrite existing values)
    if 'CVD' not in ohlcv_df.columns:
        ohlcv_df['CVD'] = 0.0
    if 'dCVD' not in ohlcv_df.columns:
        ohlcv_df['dCVD'] = 0.0
    
    # Process each candle
    cvd_cumulative = 0.0
    prev_cvd = 0.0
    
    for idx, candle in ohlcv_df.iterrows():
        candle_time = candle['ts'].floor('1min')
        candle_trades = trades_df[trades_df['candle_time'] == candle_time]
        
        if len(candle_trades) > 0:
            # Calculate volumes (using 'is_buyer_maker' from raw parquet)
            buy_trades = candle_trades[candle_trades['is_buyer_maker'] == False]  # Taker buy = aggressive buy
            sell_trades = candle_trades[candle_trades['is_buyer_maker'] == True]   # Taker sell = aggressive sell
            
            buy_vol = buy_trades['quantity'].sum() if len(buy_trades) > 0 else 0.0
            sell_vol = sell_trades['quantity'].sum() if len(sell_trades) > 0 else 0.0
            tot_vol = candle_trades['quantity'].sum()
            
            # Size statistics
            sizes = candle_trades['quantity']
            mean_size = sizes.mean() if len(sizes) > 0 else 0.0
            max_size = sizes.max() if len(sizes) > 0 else 0.0
            p95_size = sizes.quantile(0.95) if len(sizes) > 0 else 0.0
            n_trades = len(candle_trades)
            
            # Signed volume and imbalance
            signed_vol = buy_vol - sell_vol
            imb_aggr = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0.0
            
            # Block trades (large trades > 95th percentile)
            if p95_size > 0:
                block_trades = len(sizes[sizes >= p95_size])
            else:
                block_trades = 0
            
            # CVD (Cumulative Volume Delta)
            cvd_cumulative += signed_vol
            dCVD = cvd_cumulative - prev_cvd
            prev_cvd = cvd_cumulative
            
            # Signed volatility (volatility weighted by order flow)
            price_changes = candle_trades['price'].diff().abs()
            if len(price_changes) > 1 and signed_vol != 0:
                signed_volatility = price_changes.mean() * np.sign(signed_vol)
            else:
                signed_volatility = 0.0
            
            # Volume-weighted tick return
            tick_returns = candle_trades['price'].pct_change().fillna(0.0)
            qty = candle_trades['quantity'].fillna(0.0)
            vw_tick_return = (tick_returns * qty).sum() / (qty.sum() + 1e-9)

            # Realized volatility (intra-trade) within the bar
            rv = (tick_returns ** 2).sum()

            # Impact proxy
            r1 = candle['close'] / candle['open'] - 1 if candle['open'] > 0 else 0
            impact_proxy = abs(r1) / (tot_vol + 1e-9) if tot_vol > 0 else 0.0
            
            # Update dataframe
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
    
    print(f"✅ Trade features calculated for {len(ohlcv_df)} candles from raw data")
    return ohlcv_df


def load_2025_data(symbol: str = "DOGEUSDT", interval: str = "1m") -> pd.DataFrame:
    """Load all 2025 data for backtesting"""
    
    print(f"📊 Loading 2025 data for {symbol} {interval}...")
    
    # Try to load advanced features first, fall back to basic
    try:
        features_df = pq.read_table(f"features/advanced_features_{symbol}_{interval}.parquet").to_pandas()
        print(f"✅ Using advanced features: {len(features_df.columns)} columns")
    except FileNotFoundError:
        features_df = pq.read_table(f"features/features_{symbol}_{interval}.parquet").to_pandas()
        print(f"⚠️  Using basic features: {len(features_df.columns)} columns")
    
    # Filter to 2025 data only
    features_df['ts'] = pd.to_datetime(features_df['ts'])
    data_2025 = features_df[features_df['ts'].dt.year == 2025].copy()
    
    print(f"✅ 2025 data loaded: {len(data_2025):,} samples")
    print(f"   Period: {data_2025['ts'].min()} to {data_2025['ts'].max()}")
    
    return data_2025

def main():
    parser = argparse.ArgumentParser(description="Two-Phase Trading System Backtest")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--binary_model", default="Live/DOGE/models/binary.pth")
    parser.add_argument("--directional_model", default="Live/DOGE/models/directonal.pth")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--position_size", type=float, default=0.10)
    parser.add_argument("--leverage", type=float, default=25.0)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--move_threshold", type=float, default=0.3)
    parser.add_argument("--direction_threshold", type=float, default=0.55)
    parser.add_argument("--max_hold_minutes", type=int, default=30)
    parser.add_argument("--stop_loss", type=float, default=0.015)
    parser.add_argument("--take_profit", type=float, default=0.035)
    parser.add_argument("--scaler_data_dir", default="Live/DOGE", help="Directory containing scaler data CSV files (for fresh scaler fitting)")
    parser.add_argument("--live", action="store_true", help="Use live data from Live/DOGE/ directory instead of parquet files")
    parser.add_argument("--binance", action="store_true", help="Use Binance data files (binancedatadoge.csv, binanceaggtradesdoge.csv) instead of MEXC data")
    parser.add_argument("--gui", action="store_true", help="Run with live GUI")
    parser.add_argument("--raw", action="store_true", help="Use raw OHLCV and aggtrades from data_parquet/ instead of precomputed features")
    
    args = parser.parse_args()
    
    print(f"🎯 Two-Phase Trading System Backtest")
    print(f"   Symbol: {args.symbol} {args.interval}")
    print(f"   Capital: ${args.capital:,.0f}")
    print(f"   Position size: {args.position_size:.1%} per trade")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Fees: {args.fee_rate:.2%} per side ({args.fee_rate*2:.2%} round trip)")
    if args.raw:
        print(f"   🔴 RAW MODE: Computing features from raw parquet data")
    elif args.live:
        print(f"   🔴 LIVE MODE: Using CSV data from Live/DOGE/")
    
    # Load data - either live CSV files or 2025 parquet data
    if args.live:
        if args.binance:
            print("🔴 LIVE MODE: Using BINANCE CSV data from Live/DOGE/ directory")
        else:
            print("🔴 LIVE MODE: Using MEXC CSV data from Live/DOGE/ directory")
        df_2025 = load_live_data(use_binance=args.binance)
    elif args.raw:
        print("🔴 RAW MODE: Using raw parquet data from data_parquet/ directory")
        df_2025 = load_raw_data(args.symbol, args.interval)
    else:
        df_2025 = load_2025_data(args.symbol, args.interval)
    
    # Handle GUI mode differently - GUI must run on main thread on Windows
    if args.gui:
        print("🖥️  [DEBUG] GUI mode - running backtest in background thread...")
        
        # Initialize GUI on main thread
        gui = BacktestGUI()
        print("🖥️  [DEBUG] GUI initialized on main thread")
        
        # Run backtest in separate thread
        import threading
        backtest_complete = threading.Event()
        backtest_results = {}
        
        def run_backtest():
            try:
                print("🖥️  [DEBUG] Starting backtest in thread...")
                
                # Initialize backtester
                backtester = TwoPhaseBacktester(
                    binary_model_path=args.binary_model,
                    directional_model_path=args.directional_model,
                    initial_capital=args.capital,
                    position_size_pct=args.position_size,
                    leverage=args.leverage,
                    fee_rate=args.fee_rate,
                    gui=gui
                )
                
                # Run backtest
                results = backtester.run_backtest(
                    df_2025,
                    move_threshold=args.move_threshold,
                    direction_threshold=args.direction_threshold,
                    max_hold_minutes=args.max_hold_minutes,
                    stop_loss_pct=args.stop_loss,
                    take_profit_pct=args.take_profit
                )
                
                backtest_results.update(results)
                backtest_complete.set()
                print("🖥️  [DEBUG] Backtest thread completed")
                
            except Exception as e:
                print(f"🖥️  [ERROR] Backtest thread failed: {e}")
                import traceback
                traceback.print_exc()
                backtest_complete.set()
        
        # Start backtest thread
        backtest_thread = threading.Thread(target=run_backtest, daemon=True)
        backtest_thread.start()
        
        # Run GUI on main thread
        print("🖥️  [DEBUG] Starting GUI on main thread...")
        gui.run()
        
        # Wait for backtest to complete and show results
        if backtest_complete.wait(timeout=1):
            if backtest_results:
                print("\\n🎯 BACKTEST RESULTS:")
                for key, value in backtest_results.items():
                    if isinstance(value, (int, float)):
                        print(f"   {key}: {value}")
        
        return  # Exit early for GUI mode
    
    # Non-GUI mode
    gui = None
    
    # Initialize backtester
    backtester = TwoPhaseBacktester(
        binary_model_path=args.binary_model,
        directional_model_path=args.directional_model,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        leverage=args.leverage,
        fee_rate=args.fee_rate,
        gui=gui
    )
    
    # Run backtest
    results = backtester.run_backtest(
        df_2025,
        move_threshold=args.move_threshold,
        direction_threshold=args.direction_threshold,
        max_hold_minutes=args.max_hold_minutes,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        scaler_data_dir=args.scaler_data_dir
    )
    
    # Display results
    print(f"\\n🎯 BACKTEST RESULTS:")
    print(f"   Total Trades: {results['total_trades']}")
    if results['total_trades'] > 0:
        print(f"   Win Rate: {results['win_rate']:.1%} ({results['winning_trades']} wins, {results['losing_trades']} losses)")
    else:
        print(f"   Win Rate: N/A (0 trades - insufficient data for sequence length requirements)")
    print(f"   Long/Short: {results.get('long_trades', 0)}/{results.get('short_trades', 0)}")
    print(f"   Total P&L: ${results.get('total_pnl', 0):+,.2f}")
    print(f"   Total Return: {results.get('total_return', 0):+.1%}")
    print(f"   Final Capital: ${results.get('final_capital', 100000):,.2f}")
    if results['total_trades'] > 0:
        print(f"   Average Win: ${results['avg_win']:+.2f}")
        print(f"   Average Loss: ${results['avg_loss']:+.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"   Avg Duration: {results['avg_duration_minutes']:.1f} minutes")
    else:
        print(f"   Average Win: N/A")
        print(f"   Average Loss: N/A") 
        print(f"   Profit Factor: N/A")
        print(f"   Max Drawdown: N/A")
        print(f"   Avg Duration: N/A")
    
    # Final GUI update
    if gui:
        gui.queue_update({
            'type': 'progress',
            'progress': 100,
            'status': f"🎉 Backtest Complete! Total P&L: ${results['total_pnl']:+,.2f}"
        })
        
        gui.queue_update({
            'type': 'stats',
            'equity': results['final_capital'],
            'pnl': results['total_pnl'],
            'fees': results['total_fees'],
            'trades': results['total_trades'],
            'win_rate': results['win_rate'] * 100,
            'position': 'NONE'
        })
        
        print(f"\\n🖥️  GUI is running - close the window when done viewing results")
        
        # Keep main thread alive to allow GUI updates
        try:
            input("\\nPress Enter to exit...")
        except KeyboardInterrupt:
            print("\\n👋 Shutting down...")
    
    # Exit reason breakdown
    if results['total_trades'] > 0 and 'exit_reasons' in results:
        print(f"\\n📊 Exit Reasons:")
        for reason, count in results['exit_reasons'].items():
            print(f"   {reason}: {count} ({count/results['total_trades']:.1%})")
    else:
        print(f"\\n📊 Exit Reasons: N/A (no trades executed)")
    
    # Save results
    results_filename = f"backtest_results_{args.symbol}_{args.interval}_2025.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Results saved to: {results_filename}")
    
    # Performance assessment
    if results['total_return'] > 0.10:  # 10%+ return
        print(f"\\n🎉 EXCELLENT PERFORMANCE! System is profitable!")
    elif results['total_return'] > 0.05:  # 5%+ return
        print(f"\\n👍 GOOD PERFORMANCE! System shows promise!")
    elif results['total_return'] > 0:
        print(f"\\n👌 POSITIVE PERFORMANCE! System is profitable but could be better!")
    else:
        print(f"\\n❌ NEGATIVE PERFORMANCE! System needs improvement!")
    
    # Final feature completeness summary
    print(f"\\n🔍 FINAL FEATURE COMPLETENESS SUMMARY:")
    if hasattr(backtester, 'missing_binary_features'):
        if backtester.missing_binary_features == 0:
            print(f"   ✅ Binary Model: ALL 38 features computed successfully (0 missing)")
        else:
            print(f"   ⚠️  Binary Model: {backtester.missing_binary_features} features missing out of 38")
    else:
        print(f"   ❓ Binary Model: Feature count not tracked")
        
    if hasattr(backtester, 'missing_directional_features'):
        if backtester.missing_directional_features == 0:
            print(f"   ✅ Directional Model: ALL 238 features computed successfully (0 missing)")
        else:
            print(f"   ⚠️  Directional Model: {backtester.missing_directional_features} features missing out of 238")
    else:
        print(f"   ❓ Directional Model: Feature count not tracked")
    
    # 📊 EXPORT BACKTESTER FEATURES TO CSV FOR COMPARISON
    if hasattr(backtester, 'debug_features_df') and backtester.debug_features_df is not None:
        csv_path = f"backtester_features_{args.symbol}_{args.interval}_debug.csv"
        backtester.debug_features_df.to_csv(csv_path, index=True)
        print(f"📊 Backtester features exported to: {csv_path}")
    else:
        print("⚠️  No debug features to export")
    
    return results

if __name__ == "__main__":
    main()
