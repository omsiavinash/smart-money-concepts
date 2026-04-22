import os
import json
import logging
import pandas as pd
import numpy as np

from src.features.ict_features import ICTFeatures
from src.strategy.generator import ICTStrategy
from src.models.lstm_model import MultiTaskICTLSTM, ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_backtest(data_path="data/training_data.csv", config_path="config.json"):
    """
    Simulates trading over the historical dataset using the trained AI model.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found at {data_path}. Please run train.py first.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
    df.sort_index(inplace=True)

    symbols = df['symbol'].unique() if 'symbol' in df.columns else ["BTC/USDT"]

    # Init Strategy
    strategy = ICTStrategy(algorithm_version=config["trading"]["algorithm_version"])

    # Load AI Model
    seq_length = 60
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'avg_volume', 'premium_discount']
    target_cols = ['mss', 'fvg', 'ob', 'liquidity_grab', 'breaker_block', 'ote']

    input_size = len(feature_cols)
    hidden_size = config["model"]["lstm_hidden_size"]
    num_layers = config["model"]["lstm_num_layers"]

    model = MultiTaskICTLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_tasks=len(target_cols)
    )
    trainer = ModelTrainer(model, learning_rate=config["model"]["learning_rate"])

    model_path = config["model"]["model_path"]
    scaler_path = os.path.join(os.path.dirname(model_path), "scaler.json")

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        trainer.load_model(model_path)
        with open(scaler_path, "r") as f:
            scaler_data = json.load(f)
            scaler_mean = pd.Series(scaler_data["mean"])
            scaler_std = pd.Series(scaler_data["std"])
        logger.info(f"Loaded trained AI model and global scaler from {os.path.dirname(model_path)} for backtesting")
    else:
        logger.error(f"AI Model or scaler not found at {model_path}. Please run train.py first.")
        return

    # Backtest Metrics
    initial_balance = 10000.0
    balance = initial_balance
    risk_pct = 1.0

    wins = 0
    losses = 0
    total_trades = 0

    logger.info("Starting backtest simulation over historical data...")

    # We will step through the dataframe row by row, starting from seq_length
    for i in range(seq_length, len(df)):
        # Normalize the past 60 candles using the global training scaler
        window_df = df.iloc[i-seq_length:i]

        normalized_features = (window_df[feature_cols] - scaler_mean) / (scaler_std + 1e-8)

        x_seq = normalized_features.values

        # Predict ICT Concepts
        predictions = trainer.predict(x_seq)

        # Current data snapshot
        current_data = window_df.iloc[-1].to_dict()
        for j, col in enumerate(target_cols):
            current_data[col] = predictions[j]

        # Simplified HTF mock for backtest
        htf_data = {'liquidity_grab': predictions[3], 'ob': predictions[2]}

        # Generate Signal
        signal = strategy.generate_signal(current_data, htf_data)

        if signal:
            total_trades += 1
            action = signal['action']
            entry_price = current_data['close']

            # Simple TP/SL calculation for backtest
            min_rr = 3.0 if config["trading"]["algorithm_version"] == "v4" else 1.2
            if action == "BUY":
                sl = current_data.get('low', entry_price * 0.99) * 0.999
                tp = entry_price + (entry_price - sl) * min_rr
            else:
                sl = current_data.get('high', entry_price * 1.01) * 1.001
                tp = entry_price - (sl - entry_price) * min_rr

            risk_amount = balance * (risk_pct / 100)

            # Simulate Outcome: Look ahead 50 candles to see what hit first
            future_df = df.iloc[i:min(i+50, len(df))]

            outcome = "TIMED_OUT"
            for _, row in future_df.iterrows():
                if action == "BUY":
                    if row['low'] <= sl:
                        outcome = "LOSS"
                        balance -= risk_amount
                        losses += 1
                        break
                    elif row['high'] >= tp:
                        outcome = "WIN"
                        balance += (risk_amount * min_rr)
                        wins += 1
                        break
                elif action == "SELL":
                    if row['high'] >= sl:
                        outcome = "LOSS"
                        balance -= risk_amount
                        losses += 1
                        break
                    elif row['low'] <= tp:
                        outcome = "WIN"
                        balance += (risk_amount * min_rr)
                        wins += 1
                        break

    # Final Report
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    pnl = balance - initial_balance
    pnl_pct = (pnl / initial_balance) * 100

    print("\n" + "="*50)
    print("🎯 ICT-AI BOT BACKTEST REPORT")
    print("="*50)
    print(f"Total Trades Taken: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Starting Balance: ${initial_balance:,.2f}")
    print(f"Ending Balance:   ${balance:,.2f}")
    print(f"Total P/L:        ${pnl:,.2f} ({pnl_pct:.2f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_backtest()
