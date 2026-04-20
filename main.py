import json
import time
import logging
from typing import Dict, Any

from src.data.fetcher import DataFetcher
from src.features.ict_features import ICTFeatures
import os
import torch
import pandas as pd
import numpy as np
import requests
from src.strategy.generator import ICTStrategy
from src.execution.engine import ExecutionEngine
from src.models.lstm_model import MultiTaskICTLSTM, ModelTrainer, ICTDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingBotDaemon:
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.fetcher = DataFetcher(
            exchange_id=self.config["exchange"]["name"],
            testnet=False,
            api_key=self.config["exchange"].get("api_key", ""),
            api_secret=self.config["exchange"].get("api_secret", "")
        )

        # In live mode, pass the initialized ccxt exchange object to the Execution Engine
        self.strategy = ICTStrategy(algorithm_version=self.config["trading"]["algorithm_version"])
        self.execution = ExecutionEngine(
            mode=self.config["trading"]["mode"],
            risk_pct=self.config["trading"]["risk_per_trade_pct"],
            exchange=self.fetcher.exchange
        )

        # Initialize Telegram Notifier (Mocked for now, integrate python-telegram-bot later)
        self.telegram_enabled = self.config["telegram"]["enabled"]

        # Load AI Model
        self._init_ai_model()

    def _init_ai_model(self):
        """Initializes and loads the trained PyTorch LSTM model."""
        self.seq_length = 60
        # Dummy dataset to get feature cols and target cols info
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume', 'momentum', 'avg_volume', 'premium_discount']
        self.target_cols = ['mss', 'fvg', 'ob', 'liquidity_grab', 'breaker_block', 'ote']

        input_size = len(self.feature_cols)
        hidden_size = self.config["model"]["lstm_hidden_size"]
        num_layers = self.config["model"]["lstm_num_layers"]

        self.ai_model = MultiTaskICTLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_tasks=len(self.target_cols)
        )
        self.trainer = ModelTrainer(self.ai_model, learning_rate=self.config["model"]["learning_rate"])

        model_path = self.config["model"]["model_path"]
        if os.path.exists(model_path):
            self.trainer.load_model(model_path)
            logger.info(f"Loaded trained AI model from {model_path}")
        else:
            logger.warning(f"AI Model not found at {model_path}. Please run train.py first. Proceeding with deterministic logic fallback.")
            self.ai_model = None

    def _send_telegram_alert(self, message: str):
        if self.telegram_enabled:
            bot_token = self.config["telegram"].get("bot_token")
            chat_id = self.config["telegram"].get("chat_id")
            if bot_token and chat_id and bot_token != "YOUR_BOT_TOKEN":
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                try:
                    requests.post(url, json=payload, timeout=5)
                    logger.info(f"[TELEGRAM ALERT] Successfully sent message.")
                except Exception as e:
                    logger.error(f"[TELEGRAM ALERT FAILED] {e}")
            else:
                logger.info(f"[TELEGRAM ALERT MOCKED] {message}")

    def run_cycle(self):
        """
        Executes a single monitoring cycle.
        """
        symbols = self.config["trading"]["symbols"]
        base_tf = self.config["trading"]["base_timeframe"]
        htf_tfs = self.config["trading"]["htf_timeframes"]

        for symbol in symbols:
            logger.info(f"Analyzing {symbol}...")

            # 1. Fetch Multi-Timeframe Data
            all_timeframes = [base_tf] + htf_tfs
            try:
                data_dict = self.fetcher.fetch_multi_timeframe_data(symbol, all_timeframes, limit=100)
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

            if base_tf not in data_dict:
                continue

            # 2. Feature Engineering (ICT Concepts)
            base_df = data_dict[base_tf]
            features_engine = ICTFeatures(base_df)
            base_features_df = features_engine.generate_all_features()

            # Use AI Model for inference if available, otherwise use deterministic features
            current_data = base_features_df.iloc[-1].to_dict()

            if self.ai_model is not None and len(base_features_df) >= self.seq_length:
                # Normalize inputs exactly as in training
                data_mean = base_features_df[self.feature_cols].mean()
                data_std = base_features_df[self.feature_cols].std()
                normalized_features = (base_features_df[self.feature_cols] - data_mean) / (data_std + 1e-8)

                # Get the last sequence
                x_seq = normalized_features.values[-self.seq_length:]

                # Predict
                predictions = self.trainer.predict(x_seq)

                # Override deterministic labels with AI predictions
                for i, col in enumerate(self.target_cols):
                    current_data[col] = predictions[i]
                logger.debug(f"AI Predictions for {symbol}: {predictions}")

            # Calculate HTF Features
            # We aggregate HTF concepts to pass to the strategy generator
            htf_data = {}
            for htf in htf_tfs:
                if htf in data_dict:
                    htf_engine = ICTFeatures(data_dict[htf])
                    htf_df = htf_engine.generate_all_features()

                    # Store latest HTF state
                    latest_htf = htf_df.iloc[-1]
                    htf_data['liquidity_grab'] = latest_htf.get('liquidity_grab', 0)
                    htf_data['ob'] = latest_htf.get('ob', 0)
                    # We just keep the most relevant/highest timeframe state for simplicity here

            # 3. Strategy & Signal Generation
            signal = self.strategy.generate_signal(current_data, htf_data)

            if signal:
                logger.info(f"Signal generated on {symbol}: {signal}")
                self._send_telegram_alert(f"🚨 Signal Alert: {signal['action']} {symbol} (Conf: {signal['confidence']:.2f})")

                # 4. Execution & Risk Management
                # Pass recent swing levels for SL calculation
                market_data = {
                    "close": current_data["close"],
                    "recent_swing_high": current_data.get("recent_swing_high", current_data["high"] * 1.01),
                    "recent_swing_low": current_data.get("recent_swing_low", current_data["low"] * 0.99)
                }

                trade = self.execution.execute_trade(symbol, signal, market_data)
                if trade:
                    trade_msg = f"✅ Trade Executed: {trade['action']} {trade['quantity']} {symbol} @ {trade['entry_price']:.2f}\nSL: {trade['stop_loss']:.2f}\nTP: {trade['take_profit']:.2f}"
                    self._send_telegram_alert(trade_msg)

    def start(self, interval_seconds: int = 900): # 900s = 15m
        """
        Starts the continuous daemon loop.
        """
        logger.info("Starting ICT Trading Bot Daemon...")
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            logger.info(f"Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)

if __name__ == "__main__":
    bot = TradingBotDaemon()
    # For testing, we run one cycle and exit. In production, call bot.start()
    bot.run_cycle()
