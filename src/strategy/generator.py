import pandas as pd
from typing import Dict, Any, Optional

class ICTStrategy:
    def __init__(self, algorithm_version: str = "v4"):
        self.version = algorithm_version

    def generate_signal(self, current_data: Dict[str, Any], htf_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generates a trading signal based on the specified algorithm version.

        current_data: Dictionary containing the latest predicted/calculated ICT features for the base timeframe.
        htf_data: Dictionary containing HTF features (e.g. HTF Liquidity Grabs, POI).
        """
        if self.version == "v4":
            return self._v4_algorithm(current_data, htf_data)
        elif self.version == "v3":
            return self._v3_algorithm(current_data, htf_data)
        else:
            raise ValueError(f"Unknown algorithm version: {self.version}")

    def _v4_algorithm(self, current: Dict[str, Any], htf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        V4 5-Tier Confirmation System
        Mandatory: MSS (Tier 1), Risk-Reward Ratio >= 1:3.0 (Tier 5 - validated at execution)
        Optional: HTF Liquidity Grab, POI, FVG
        Requires: Both mandatory + at least 1 optional (3/5 total).
        """
        # Determine Direction from MSS
        direction = 0
        if current.get('mss') == 1:
            direction = 1 # Bullish
        elif current.get('mss') == -1:
            direction = -1 # Bearish

        if direction == 0:
            return None # Mandatory MSS not met

        # Check optional tiers
        confirmations = 0
        reasons = [f"Mandatory: {'Bullish' if direction == 1 else 'Bearish'} Market Structure Shift"]

        # 1. FVG presence in the same direction
        if current.get('fvg') == direction:
            confirmations += 1
            reasons.append("Optional: Fair Value Gap aligned")

        # 2. HTF Liquidity Grab in the same direction
        if htf.get('liquidity_grab') == direction:
            confirmations += 1
            reasons.append("Optional: HTF Liquidity Grab aligned")

        # 3. POI / Order Block in the same direction
        if current.get('ob') == direction or htf.get('ob') == direction:
            confirmations += 1
            reasons.append("Optional: Order Block / POI aligned")

        # Must have at least 1 optional confirmation
        if confirmations >= 1:
            # We have a valid setup. Return signal intent.
            # Entry/SL/TP and R:R checks will be finalised in Execution Engine.
            return {
                "action": "BUY" if direction == 1 else "SELL",
                "confidence": 0.7 + (0.1 * confirmations), # pseudo-confidence score
                "version": "v4",
                "tier_confirmations": confirmations + 2, # +2 for MSS and R:R
                "reasons": reasons
            }

        return None

    def _v3_algorithm(self, current: Dict[str, Any], htf: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        V3 Algorithm (Production 75% Win Rate)
        Mandatory: MSS
        Requires Risk-Reward >= 1:1.2
        Confidence Threshold: 62%
        Requires 3 of 5 overall confirmations.
        """
        # simplified fallback for V3
        direction = current.get('mss', 0)
        if direction == 0:
            return None

        confirmations = 0
        reasons = [f"Mandatory: {'Bullish' if direction == 1 else 'Bearish'} Market Structure Shift"]

        if current.get('fvg') == direction:
            confirmations += 1
            reasons.append("Fair Value Gap")
        if current.get('ob') == direction:
            confirmations += 1
            reasons.append("Order Block")
        if current.get('liquidity_grab') == direction:
            confirmations += 1
            reasons.append("Liquidity Grab")
        if current.get('premium_discount') == (1 if direction == 1 else -1):
            confirmations += 1
            reasons.append("Premium/Discount alignment")

        if confirmations >= 2: # MSS (1) + 2 others = 3/5
            return {
                "action": "BUY" if direction == 1 else "SELL",
                "confidence": 0.62 + (0.05 * confirmations),
                "version": "v3",
                "tier_confirmations": confirmations + 1,
                "reasons": reasons
            }

        return None
