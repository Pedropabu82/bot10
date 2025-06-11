import pandas as pd
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, CCIIndicator, ROCIndicator
from ta.volatility import (
    BollingerBands,
    AverageTrueRange,
    KeltnerChannel,
    DonchianChannel,
)
from ta.volume import (
    OnBalanceVolumeIndicator,
    ChaikinMoneyFlowIndicator,
    VolumeWeightedAveragePrice,
)
import numpy as np


def add_indicators(df: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """Add technical indicators columns based on parameter configuration."""

    df = df.copy()

    # Moving averages
    ema_fast_period = parameters.get("ema_fast", 12)
    ema_slow_period = parameters.get("ema_slow", 26)
    df["ema_fast"] = EMAIndicator(df["close"], window=ema_fast_period).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], window=ema_slow_period).ema_indicator()

    macd = MACD(
        df["close"],
        window_fast=parameters.get("macd_fast", 12),
        window_slow=parameters.get("macd_slow", 26),
        window_sign=parameters.get("macd_signal", 9),
    )
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ADXIndicator(
        df["high"], df["low"], df["close"], window=parameters.get("adx_window", 14)
    )
    df["adx"] = adx.adx()

    rsi = RSIIndicator(df["close"], window=parameters.get("rsi_window", 14))
    df["rsi"] = rsi.rsi()

    stoch = StochasticOscillator(
        df["high"],
        df["low"],
        df["close"],
        window=parameters.get("stoch_window", 14),
        smooth_window=parameters.get("stoch_smooth", 3),
    )
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    cci = CCIIndicator(
        df["high"], df["low"], df["close"], window=parameters.get("cci_window", 20)
    )
    df["cci"] = cci.cci()

    roc = ROCIndicator(df["close"], window=parameters.get("roc_window", 12))
    df["roc"] = roc.roc()

    bb = BollingerBands(df["close"], window=parameters.get("bb_window", 20), window_dev=parameters.get("bb_std", 2))
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_lower"] = bb.bollinger_lband()

    atr = AverageTrueRange(
        df["high"], df["low"], df["close"], window=parameters.get("atr_window", 14)
    )
    df["atr"] = atr.average_true_range()

    kelt = KeltnerChannel(
        df["high"],
        df["low"],
        df["close"],
        window=parameters.get("kelt_window", 20),
        window_atr=parameters.get("kelt_atr", 10),
    )
    df["kelt_upper"] = kelt.keltner_channel_hband()
    df["kelt_lower"] = kelt.keltner_channel_lband()

    donch = DonchianChannel(
        df["high"], df["low"], window=parameters.get("donch_window", 20)
    )
    df["donch_upper"] = donch.donchian_channel_hband()
    df["donch_lower"] = donch.donchian_channel_lband()

    obv = OnBalanceVolumeIndicator(df["close"], df["volume"])
    df["obv"] = obv.on_balance_volume()

    cmf = ChaikinMoneyFlowIndicator(
        df["high"], df["low"], df["close"], df["volume"], window=parameters.get("cmf_window", 20)
    )
    df["cmf"] = cmf.chaikin_money_flow()

    vwap = VolumeWeightedAveragePrice(
        df["high"], df["low"], df["close"], df["volume"], window=parameters.get("vwap_window", 20)
    )
    df["vwap"] = vwap.volume_weighted_average_price()

    # Simple SuperTrend implementation
    def _supertrend(source_df, period=10, multiplier=3):
        atr_val = AverageTrueRange(
            source_df["high"], source_df["low"], source_df["close"], window=period
        ).average_true_range()
        hl2 = (source_df["high"] + source_df["low"]) / 2
        upperband = hl2 + multiplier * atr_val
        lowerband = hl2 - multiplier * atr_val

        final_upper = upperband.copy()
        final_lower = lowerband.copy()
        for i in range(1, len(source_df)):
            if source_df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
                final_upper.iloc[i] = max(upperband.iloc[i], final_upper.iloc[i - 1])
            else:
                final_upper.iloc[i] = upperband.iloc[i]

            if source_df["close"].iloc[i - 1] < final_lower.iloc[i - 1]:
                final_lower.iloc[i] = min(lowerband.iloc[i], final_lower.iloc[i - 1])
            else:
                final_lower.iloc[i] = lowerband.iloc[i]

        supertrend = pd.Series(np.nan, index=source_df.index)
        direction = pd.Series(1, index=source_df.index)
        for i in range(1, len(source_df)):
            prev_super = supertrend.iloc[i - 1]
            if source_df["close"].iloc[i] > final_upper.iloc[i - 1]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            elif source_df["close"].iloc[i] < final_lower.iloc[i - 1]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = prev_super
                direction.iloc[i] = direction.iloc[i - 1]
        return supertrend, direction

    st_period = parameters.get("supertrend_window", 10)
    st_mult = parameters.get("supertrend_mult", 3)
    st, st_dir = _supertrend(df, st_period, st_mult)
    df["supertrend"] = st
    df["supertrend_dir"] = st_dir

    return df


def extract_features(
    df: pd.DataFrame,
    **parameters,
) -> pd.DataFrame:
    """Compute model features from OHLCV dataframe using ``add_indicators``.

    Parameters
    ----------
    df : pd.DataFrame
        Candle data with OHLCV columns.
    parameters : dict
        Indicator windows and thresholds.
    """

    features = add_indicators(df, parameters)
    features["volume"] = df["volume"]
    return features.dropna().reset_index(drop=True)
