import websocket
import json
import time
import signal
import sys
import threading
import csv
import os
import requests
from datetime import datetime, timedelta

# =================== CONFIGURATION ===================
SYMBOL = "DOGEUSDT"  # DOGE/USDT futures symbol (Binance format)
# =====================================================

# Binance Futures WebSocket endpoints
WS_URL = "wss://fstream.binance.com/ws/"
INTERVAL = "1m"

# Auto-generate CSV filenames based on symbol (lowercase, remove USDT)
def get_csv_filenames(symbol):
    if not symbol:
        return "binancedata.csv", "binanceaggtrades.csv"
    # Remove USDT and convert to lowercase
    clean_symbol = symbol.replace('USDT', '').replace('usdt', '').lower()
    return f"binancedata{clean_symbol}.csv", f"binanceaggtrades{clean_symbol}.csv"

CSV_FILENAME, AGGTRADES_FILENAME = get_csv_filenames(SYMBOL)

# Global variables for WebSocket data
latest_kline = None
latest_trade = None
last_candle_time = None
last_trade_id = None
ws_kline = None
ws_trade = None
heartbeat_timer = None
is_running = True

# Historical data fetching removed - we can't get historical aggtrades anyway

def save_to_csv(timestamp, open_price, high_price, low_price, close_price, volume):
    """Save candle data to CSV file"""
    try:
        # Convert timestamp to readable datetime
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(CSV_FILENAME)
        
        with open(CSV_FILENAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['Timestamp', 'DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                print(f"📄 Created new CSV file: {CSV_FILENAME}")
            
            # Write candle data
            writer.writerow([
                timestamp,
                dt.strftime('%Y-%m-%d %H:%M:%S'),
                open_price,
                high_price,
                low_price,
                close_price,
                volume
            ])
            
    except Exception as e:
        print(f"⚠️ CSV save error: {e}")

def save_trade_to_csv(trade_id, price, quantity, timestamp, is_buyer_maker):
    """Save aggregate trade data to CSV file"""
    try:
        # Convert timestamp to readable datetime
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(AGGTRADES_FILENAME)
        
        with open(AGGTRADES_FILENAME, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header if file is new
            if not file_exists:
                writer.writerow(['TradeID', 'Price', 'Quantity', 'Timestamp', 'DateTime', 'IsBuyerMaker'])
                print(f"📄 Created new aggTrades CSV file: {AGGTRADES_FILENAME}")
            
            # Write trade data
            writer.writerow([
                trade_id,
                price,
                quantity,
                timestamp,
                dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],  # Include milliseconds
                is_buyer_maker
            ])
            
    except Exception as e:
        print(f"⚠️ AggTrades CSV save error: {e}")

def on_message(ws, message):
    global latest_kline, latest_trade, last_candle_time, last_trade_id
    try:
        # Debug: Print raw message occasionally
        if hasattr(on_message, 'debug_count'):
            on_message.debug_count += 1
        else:
            on_message.debug_count = 1
            
        if on_message.debug_count % 100 == 1:  # Print every 100th message for debugging
            print(f"🔍 RAW MESSAGE: {message[:200]}...")
            
        data = json.loads(message)
        
        if on_message.debug_count % 100 == 1:  # Print parsed data
            print(f"🔍 PARSED DATA: Type: {type(data)}, Content: {str(data)[:200]}...")
        
        # Handle kline data
        if data.get('e') == 'kline':  # Binance kline event
            kline_data = data.get('k', {})
            if kline_data.get('s') == SYMBOL:
                # Extract OHLCV data from Binance WebSocket message
                # Format: {"t": start_time, "T": end_time, "s": symbol, "i": interval, "o": open, "c": close, "h": high, "l": low, "v": volume, "x": is_closed}
                timestamp = int(kline_data.get('T', 0))  # End time in milliseconds
                open_price = float(kline_data.get('o', 0))
                high_price = float(kline_data.get('h', 0))
                low_price = float(kline_data.get('l', 0))
                close_price = float(kline_data.get('c', 0))
                volume = float(kline_data.get('v', 0))
                is_closed = kline_data.get('x', False)  # True when kline is closed
                
                kline_array = [timestamp, open_price, high_price, low_price, close_price, volume]
                
                latest_kline = {
                    'source': 'binance-futures-websocket',
                    'data': kline_array
                }
                
                # Only print and save when we get a CLOSED candle (different timestamp)
                if is_closed and timestamp != last_candle_time:
                    print(f"🕒 NEW CANDLE: {ts_to_time(timestamp)} | O:{open_price} H:{high_price} L:{low_price} C:{close_price} V:{volume}")
                    
                    # Save to CSV
                    save_to_csv(timestamp, open_price, high_price, low_price, close_price, volume)
                    print(f"💾 Saved candle to {CSV_FILENAME}")
                    
                    last_candle_time = timestamp
        
        # Handle aggregate trade data
        elif data.get('e') == 'aggTrade':  # Binance aggTrade event
            if data.get('s') == SYMBOL:
                # Extract trade data from Binance WebSocket message
                # Format: {"e": "aggTrade", "E": event_time, "s": symbol, "a": agg_trade_id, "p": price, "q": quantity, "f": first_trade_id, "l": last_trade_id, "T": trade_time, "m": is_buyer_maker}
                trade_id = int(data.get('a', 0))  # Aggregate trade ID
                price = float(data.get('p', 0))
                quantity = float(data.get('q', 0))
                timestamp = int(data.get('T', 0))  # Trade time in milliseconds
                is_buyer_maker = bool(data.get('m', False))  # True if buyer is maker
                
                latest_trade = {
                    'trade_id': trade_id,
                    'price': price,
                    'quantity': quantity,
                    'timestamp': timestamp,
                    'is_buyer_maker': is_buyer_maker
                }
                
                # Save all trades (but only print every 10th trade to reduce spam)
                if on_message.debug_count % 10 == 0:
                    print(f"💱 NEW TRADE: {ts_to_time(timestamp)} | ID:{trade_id} P:{price:.6f} Q:{quantity:.2f} Side:{'MAKER' if is_buyer_maker else 'TAKER'}")
                
                # Save to aggTrades CSV
                save_trade_to_csv(trade_id, price, quantity, timestamp, is_buyer_maker)
                
                if on_message.debug_count % 10 == 0:
                    print(f"💾 Saved trade to {AGGTRADES_FILENAME}")
        
    except Exception as e:
        print(f"⚠️ WebSocket message error: {e}")
        # Don't crash on individual message errors, just log and continue
        try:
            if 'data' in locals():
                print(f"🔍 ERROR DEBUG: Data content: {str(data)[:200]}...")
        except:
            pass

def on_error(ws, error):
    print(f"🔴 WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    global heartbeat_timer, is_running, ws_kline, ws_trade
    print(f"🔌 WebSocket connection closed: {close_status_code} - {close_msg}")
    
    # Stop heartbeat timer if connection closes
    if heartbeat_timer:
        heartbeat_timer.cancel()
        heartbeat_timer = None
    
    # Only auto-reconnect if we're still running and it's not a normal close
    if is_running and close_status_code != 1000:
        print("🔄 Auto-reconnecting in 10 seconds...")
        time.sleep(10)
        start_websockets()

def send_heartbeat():
    """Send ping to keep connections alive"""
    global ws_kline, ws_trade, heartbeat_timer
    try:
        # Binance doesn't require explicit ping for market data streams
        # The ping_interval in run_forever handles this automatically
        print("💓 Heartbeat check")
        heartbeat_timer = threading.Timer(30.0, send_heartbeat)
        heartbeat_timer.start()
    except Exception as e:
        print(f"⚠️ Heartbeat error: {e}")

def on_open(ws):
    print(f"🟢 WebSocket connected - receiving {SYMBOL} kline data...")
    print(f"📡 Receiving {SYMBOL} {INTERVAL} klines")
    
    # Start heartbeat to keep connection alive
    send_heartbeat()

def start_kline_websocket():
    """Start WebSocket connection for kline data"""
    global ws_kline
    websocket.enableTrace(False)
    
    kline_stream = f"{SYMBOL.lower()}@kline_{INTERVAL}"
    ws_url = f"{WS_URL}{kline_stream}"
    print(f"🔗 Connecting to klines: {ws_url}")
    
    ws_kline = websocket.WebSocketApp(ws_url,
                                     on_open=on_open,
                                     on_message=on_message,
                                     on_error=on_error,
                                     on_close=on_close)
    
    try:
        ws_kline.run_forever(ping_interval=30, ping_timeout=10)
    except Exception as e:
        print(f"⚠️ Kline WebSocket error: {e}")
        if is_running:
            print("🔄 Retrying kline connection...")
            time.sleep(5)
            start_kline_websocket()

def start_trade_websocket():
    """Start WebSocket connection for trade data"""
    global ws_trade
    websocket.enableTrace(False)
    
    trade_stream = f"{SYMBOL.lower()}@aggTrade"
    ws_url = f"{WS_URL}{trade_stream}"
    print(f"🔗 Connecting to trades: {ws_url}")
    
    ws_trade = websocket.WebSocketApp(ws_url,
                                     on_open=on_open,
                                     on_message=on_message,
                                     on_error=on_error,
                                     on_close=on_close)
    
    try:
        ws_trade.run_forever(ping_interval=30, ping_timeout=10)
    except Exception as e:
        print(f"⚠️ Trade WebSocket error: {e}")
        if is_running:
            print("🔄 Retrying trade connection...")
            time.sleep(5)
            start_trade_websocket()

def start_websockets():
    """Start both WebSocket connections"""
    print("🚀 Starting both WebSocket connections...")
    
    # Start kline WebSocket in a separate thread
    kline_thread = threading.Thread(target=start_kline_websocket, daemon=True)
    kline_thread.start()
    
    # Start trade WebSocket in a separate thread
    trade_thread = threading.Thread(target=start_trade_websocket, daemon=True)
    trade_thread.start()
    
    # Wait for both threads
    kline_thread.join()
    trade_thread.join()

def get_latest_kline():
    return latest_kline

def signal_handler(sig, frame):
    global heartbeat_timer, ws_kline, ws_trade, is_running
    print("\n🛑 Shutting down gracefully...")
    
    # Stop auto-reconnection
    is_running = False
    
    # Stop heartbeat timer
    if heartbeat_timer:
        heartbeat_timer.cancel()
        
    # Close WebSocket connections
    if ws_kline:
        ws_kline.close()
    if ws_trade:
        ws_trade.close()
        
    sys.exit(0)

def ts_to_time(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000).strftime('%H:%M:%S')

signal.signal(signal.SIGINT, signal_handler)

print("🚀 Starting Binance Futures WebSocket data collector...")
print(f"📊 Symbol: {SYMBOL}")
print(f"💾 Saving klines to: {CSV_FILENAME}")
print(f"💾 Saving aggTrades to: {AGGTRADES_FILENAME}")
print()

print("🎯 Starting real-time data collection...")
print("Press Ctrl+C to stop")

# Start WebSockets in background thread
ws_thread = threading.Thread(target=start_websockets, daemon=True)
ws_thread.start()

print("⏳ Connecting to Binance WebSockets...")

# Keep the main thread alive - WebSockets will handle all events
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    signal_handler(None, None)
