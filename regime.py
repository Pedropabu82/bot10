import pandas as pd


def detect_regime(df: pd.DataFrame, params: dict) -> str:
    """Classify market regime given thresholds in params."""
    if df.empty:
        return "range"

    adx_th = params.get("adx_thresh", 25)
    vol_th = params.get("atr_thresh", 1.0)
    last_adx = df.get("adx").iloc[-1]
    last_atr = df.get("atr").iloc[-1]

    if last_adx > adx_th:
        return "trend"
    if last_atr > vol_th:
        return "volatile"
    return "range"

