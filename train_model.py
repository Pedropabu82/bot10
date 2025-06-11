import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(symbol: str = "BTCUSDT", model_output: str = "models/entry_model.pkl"):
    os.makedirs("models", exist_ok=True)
    path = f"data/master_features_signals_{symbol}.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    feature_cols = [c for c in df.columns if c not in ["timestamp", "entry_signal", "exit_signal"]]
    X = df[feature_cols]
    y = df["entry_signal"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_output)
    print(f"Model saved to {model_output}")


if __name__ == "__main__":
    train_model()
