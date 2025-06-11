import glob
import os
import pandas as pd


def build_master_ohlcv(symbol: str):
    files = sorted(glob.glob(f"data/ohlcv_{symbol}_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(dfs)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(f"data/master_ohlcv_{symbol}.csv", index=False)
    return df


def build_features_signals(symbol: str):
    files = sorted(glob.glob(f"data/features_signals_{symbol}_*.csv"))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_csv(f, parse_dates=["timestamp"]) for f in files]
    df = pd.concat(dfs)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(f"data/master_features_signals_{symbol}.csv", index=False)
    return df


if __name__ == "__main__":
    symbol = "BTCUSDT"
    build_master_ohlcv(symbol)
    build_features_signals(symbol)
