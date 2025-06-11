import pandas as pd
import numpy as np


def walk_forward(df: pd.DataFrame, train_size: int, test_size: int, n_splits: int):
    """Yield train/test splits for walk-forward validation."""
    max_end = train_size + test_size * n_splits
    if len(df) < max_end:
        raise ValueError("Data too short for requested splits")
    for i in range(n_splits):
        start = i * test_size
        train = df.iloc[start : start + train_size]
        test = df.iloc[start + train_size : start + train_size + test_size]
        yield train, test


def monte_carlo(df: pd.DataFrame, sample_size: int, runs: int = 100):
    """Generator yielding random contiguous samples for Monte Carlo backtesting."""
    if len(df) < sample_size:
        raise ValueError("Sample size larger than data")
    for _ in range(runs):
        start = np.random.randint(0, len(df) - sample_size)
        yield df.iloc[start : start + sample_size]
