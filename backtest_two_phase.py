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
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pyarrow.parquet as pq
from typing import Dict, List, Tuple
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
            
            # Use ARGMAX for predictions (same as training evaluation)
            move_prediction = np.argmax(move_logits.cpu().numpy(), axis=1)[0]
            
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
        
        print(f"📈 ENTER {direction}: Price={price:.6f}, Size=${position_value:,.0f}, Fees=${entry_fees:.2f}, Confidence={confidence:.3f}")
    
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
        
        print(f"📉 EXIT {self.position}: Price={price:.6f}, P&L=${net_pnl:+.2f}, Duration={trade_duration:.1f}min, Reason={reason}")
        
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
                    take_profit_pct: float = 0.035) -> Dict:
        """
        Run the two-phase backtest on the provided data.
        
        Args:
            df: DataFrame with OHLC data and features
            move_threshold: Minimum probability for Phase 1 move detection (0-1)
            direction_threshold: Minimum confidence for Phase 2 direction (0-1)
            max_hold_minutes: Maximum time to hold a position
            stop_loss_pct: Stop loss percentage (e.g., 0.015 = 1.5%)
            take_profit_pct: Take profit percentage (e.g., 0.035 = 3.5%)
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
        
        # Use fresh scaling on recent data (same as labeling script)
        from train_baseline import prepare_features
        
        latest_date = df['ts'].max()
        six_months_ago = latest_date - pd.DateOffset(months=6)
        df_train_recent = binary_features_df[df['ts'] >= six_months_ago].copy()
        
        # Apply prepare_features with fresh scaling
        train_features, fitted_scaler = prepare_features(df_train_recent, binary_feature_cols, scaler=None, fit_scaler=True)
        binary_features_scaled, _ = prepare_features(binary_features_df, binary_feature_cols, scaler=fitted_scaler, fit_scaler=False)
        
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
        
        # For directional model: create features matching the scaler's expected input
        directional_features_ordered = pd.DataFrame(index=df.index)
        
        missing_count = 0
        # Add features in the exact order the scaler expects
        for expected_col in self.directional_feature_cols:
            if expected_col in df.columns:
                directional_features_ordered[expected_col] = df[expected_col]
            else:
                # Add missing features as zeros (model should handle this gracefully)
                directional_features_ordered[expected_col] = 0.0
                missing_count += 1
                
        if missing_count > 0:
            print(f"⚠️  Added {missing_count} missing directional features as zeros")
        
        # Apply same preprocessing as training
        volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                          'max_size', 'p95_size', 'signed_vol', 'dCVD', 'CVD']
        
        for col in directional_features_ordered.columns:
            if col in volume_features:
                directional_features_ordered[col] = directional_features_ordered[col] / 1e6
            elif directional_features_ordered[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                p1, p99 = np.percentile(directional_features_ordered[col], [1, 99])
                directional_features_ordered[col] = directional_features_ordered[col].clip(p1, p99)
        
        # Fill NaN values
        directional_features_ordered = directional_features_ordered.fillna(method='ffill').fillna(0)
        
        # Apply exact scalers with matching feature names
        directional_features_scaled = self.directional_robust_scaler.transform(directional_features_ordered)
        directional_features_final = self.directional_standard_scaler.transform(directional_features_scaled)
        
        print(f"✅ Features prepared for both phases")
        
        # Main backtesting loop
        total_bars = len(df)
        start_bar = max(self.binary_sequence_length, self.directional_sequence_length)
        
        for i in range(start_bar, total_bars):
            current_row = df.iloc[i]
            current_price = current_row['close']
            current_time = current_row['ts']
            
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
                # Time-based exit
                time_in_position = (current_time - self.entry_time).total_seconds() / 60
                if time_in_position >= max_hold_minutes:
                    self.exit_position(current_price, current_time, "TIME_LIMIT")
                    continue
                
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
            
            # Only look for new entries if not in position
            if self.position is None:
                # Phase 1: Check if move is detected using EXACT same method as labeling script
                binary_sequence = binary_features_final[i-self.binary_sequence_length+1:i+1]
                move_prediction = self.predict_move(binary_sequence)
                
                if move_prediction == 1:  # Move detected (same as labeling script)
                    # Phase 2: Get direction prediction using ARGMAX (no threshold!)
                    directional_sequence = directional_features_final[i-self.directional_sequence_length+1:i+1]
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

def load_live_data() -> pd.DataFrame:
    """Load live data from CSV files in Live/DOGE/ directory"""
    import os
    
    print(f"🔴 Loading LIVE data from CSV files...")
    
    # Load OHLCV data
    ohlcv_path = "Live/DOGE/datadoge.csv"
    if not os.path.exists(ohlcv_path):
        raise FileNotFoundError(f"Live OHLCV data not found: {ohlcv_path}")
    
    ohlcv_df = pd.read_csv(ohlcv_path)
    print(f"✅ Loaded OHLCV data: {len(ohlcv_df)} candles")
    
    # Load aggregate trades data
    trades_path = "Live/DOGE/aggtradesdoge.csv"
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

    # Fill NaNs and infinities
    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    # Apply same preprocessing as live script
    volume_features = ['volume', 'quote_volume', 'buy_vol', 'sell_vol', 'tot_vol', 
                      'max_size', 'p95_size', 'mean_size', 'signed_vol', 'dCVD', 'CVD', 'signed_volatility']
    
    print(f"🔧 Applying preprocessing (scaling volume features by 1e6)...")
    for col in df.columns:
        if col in volume_features:
            df[col] = df[col] / 1e6
            print(f"   Scaled {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")
        elif col == 'impact_proxy':
            # Set impact_proxy to 0 to match training data (as done in live script)
            df[col] = 0.0
            print(f"   Set {col} to 0 (matching training data)")
        elif df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            # Apply outlier clipping
            p1, p99 = np.percentile(df[col].dropna(), [1, 99])
            df[col] = df[col].clip(p1, p99)
    
    print(f"✅ Live data processed: {len(df)} samples with {len(df.columns)} features")
    print(f"   Period: {df['ts'].min()} to {df['ts'].max()}")
    print(f"   Feature ranges after preprocessing:")
    for col in ['r1', 'volume', 'buy_vol', 'sell_vol', 'impact_proxy', 'signed_volatility']:
        if col in df.columns:
            print(f"     {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")
    
    return df


def calculate_trade_features_for_backtest(ohlcv_df: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trade-based features for each candle using aggregate trades data"""
    
    print(f"🔧 Calculating trade features from {len(trades_df)} trades...")
    
    # Group trades by minute candles
    trades_df['candle_time'] = trades_df['timestamp'].dt.floor('1min')
    
    print(f"🔧 Debug: Trade time range: {trades_df['timestamp'].min()} to {trades_df['timestamp'].max()}")
    print(f"🔧 Debug: OHLCV time range: {ohlcv_df['timestamp'].min()} to {ohlcv_df['timestamp'].max()}")
    print(f"🔧 Debug: Unique candle times in trades: {len(trades_df['candle_time'].unique())}")
    print(f"🔧 Debug: Sample trade candle times: {trades_df['candle_time'].head(3).tolist()}")
    print(f"🔧 Debug: Sample OHLCV times: {ohlcv_df['timestamp'].head(3).tolist()}")
    
    # Initialize trade feature columns
    trade_features = {
        'buy_vol': 0.0, 'sell_vol': 0.0, 'tot_vol': 0.0,
        'mean_size': 0.0, 'max_size': 0.0, 'p95_size': 0.0, 'n_trades': 0,
        'signed_vol': 0.0, 'imb_aggr': 0.0, 'block_trades': 0.0,
        'CVD': 0.0, 'dCVD': 0.0, 'signed_volatility': 0.0, 'impact_proxy': 0.0,
        'vw_tick_return': 0.0, 'rv': 0.0
    }
    
    for col in trade_features:
        ohlcv_df[col] = trade_features[col]
    
    # Process each candle
    cvd_cumulative = 0.0
    prev_cvd = 0.0
    
    for idx, candle in ohlcv_df.iterrows():
        candle_time = candle['timestamp'].floor('1min')
        candle_trades = trades_df[trades_df['candle_time'] == candle_time]
        
        if len(candle_trades) > 0:
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
    
    print(f"✅ Trade features calculated for {len(ohlcv_df)} candles")
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
    parser.add_argument("--binary_model", default="models/best_binary_DOGEUSDT_1m.pth")
    parser.add_argument("--directional_model", default="models/best_phase2_directional_DOGEUSDT_1m.pth")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--position_size", type=float, default=0.10)
    parser.add_argument("--leverage", type=float, default=25.0)
    parser.add_argument("--fee_rate", type=float, default=0.0003)
    parser.add_argument("--move_threshold", type=float, default=0.7)
    parser.add_argument("--direction_threshold", type=float, default=0.55)
    parser.add_argument("--max_hold_minutes", type=int, default=30)
    parser.add_argument("--stop_loss", type=float, default=0.015)
    parser.add_argument("--take_profit", type=float, default=0.035)
    parser.add_argument("--gui", action="store_true", help="Run with live GUI")
    parser.add_argument("--live", action="store_true", help="Use live data from Live/DOGE/ directory instead of parquet files")
    
    args = parser.parse_args()
    
    print(f"🎯 Two-Phase Trading System Backtest")
    print(f"   Symbol: {args.symbol} {args.interval}")
    print(f"   Capital: ${args.capital:,.0f}")
    print(f"   Position size: {args.position_size:.1%} per trade")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Fees: {args.fee_rate:.2%} per side ({args.fee_rate*2:.2%} round trip)")
    
    # Load data - either live CSV files or 2025 parquet data
    if args.live:
        print("🔴 LIVE MODE: Using CSV data from Live/DOGE/ directory")
        df_2025 = load_live_data()
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
        take_profit_pct=args.take_profit
    )
    
    # Display results
    print(f"\\n🎯 BACKTEST RESULTS:")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1%} ({results['winning_trades']} wins, {results['losing_trades']} losses)")
    print(f"   Long/Short: {results['long_trades']}/{results['short_trades']}")
    print(f"   Total P&L: ${results['total_pnl']:+,.2f}")
    print(f"   Total Return: {results['total_return']:+.1%}")
    print(f"   Final Capital: ${results['final_capital']:,.2f}")
    print(f"   Average Win: ${results['avg_win']:+.2f}")
    print(f"   Average Loss: ${results['avg_loss']:+.2f}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"   Avg Duration: {results['avg_duration_minutes']:.1f} minutes")
    
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
    print(f"\\n📊 Exit Reasons:")
    for reason, count in results['exit_reasons'].items():
        print(f"   {reason}: {count} ({count/results['total_trades']:.1%})")
    
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
    
    return results

if __name__ == "__main__":
    main()
