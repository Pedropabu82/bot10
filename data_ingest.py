import os
import pandas as pd


def build_master_ohlcv(symbol: str):
    file = f"data/ohlcv_{symbol}.parquet"
    if not os.path.exists(file):
        return pd.DataFrame()
    df = pd.read_parquet(file)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(f"data/master_ohlcv_{symbol}.csv", index=False)
    return df


def build_features_signals(symbol: str):
    file = f"data/features_signals_{symbol}.csv"
    if not os.path.exists(file):
        return pd.DataFrame()
    df = pd.read_csv(file, parse_dates=["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df.to_csv(f"data/master_features_signals_{symbol}.csv", index=False)
    return df


if __name__ == "__main__":
    symbol = "BTCUSDT"
    build_master_ohlcv(symbol)
    build_features_signals(symbol)
