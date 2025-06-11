import pandas as pd


def detect_regime(df: pd.DataFrame, adx_thresh: float = 25, vol_thresh: float = 1.0) -> str:
    """Classify market regime based on ADX and volatility (ATR)."""
    if df.empty:
        return "unknown"

    last_adx = df["adx"].iloc[-1]
    last_atr = df["atr"].iloc[-1]

    if last_adx > adx_thresh:
        return "trend"
    if last_atr > vol_thresh:
        return "volatile"
    return "range"
