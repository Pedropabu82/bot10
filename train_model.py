import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from features import extract_features

logger = logging.getLogger(__name__)


def train_entry_model(
    symbol: str = "BTCUSDT",
    master_path_template: str = "data/master_features_signals_{symbol}.csv",
    model_output: str = "models/entry_model.pkl"
):
    """Treina um RandomForest para sinal de entry usando o CSV de features+sinais."""
    os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
    path = master_path_template.format(symbol=symbol)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {path} not found")

    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    feature_cols = [c for c in df.columns if c not in ["entry_signal", "exit_signal"]]
    X = df[feature_cols]
    y = df["entry_signal"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, model_output)
    print(f"Entry model saved to {model_output}")


def train_log_model(
    log_path: str = "data/trade_log.csv",
    model_output: str = "models/log_xgb_model.pkl",
    config_file: str = "config.json"
):
    """Treina um XGBoost para prever win/loss a partir do log de trades."""
    os.makedirs(os.path.dirname(model_output) or ".", exist_ok=True)
    trades = pd.read_csv(log_path).dropna()
    with open(config_file, 'r') as f:
        cfg = json.load(f)

    # Filtrar só as entradas
    trades = trades[trades['type'] == 'ENTRY']

    X_list, y_list = [], []
    for _, row in trades.iterrows():
        try:
            # Recria mini-DataFrame para extrair features históricas
            df = pd.DataFrame({
                'open': [row['open']],
                'high': [row['high']],
                'low': [row['low']],
                'close': [row['close']],
                'volume': [row['volume']],
            })
            df = pd.concat([df] * 150, ignore_index=True)  # simular série temporal

            # Parâmetros de indicadores para esse symbol (opcional)
            indicator_cfg = cfg.get('indicators', {}).get(row['symbol'], {})

            feats = extract_features(
                df,
                bb_window=cfg.get('bb_period', 20),
                bb_dev=cfg.get('bb_k', 2),
                stoch_window=cfg.get('stoch_k_period', 14),
                stoch_smooth=cfg.get('stoch_d_period', 3),
                ema_fast=indicator_cfg.get('ema_short', 9),
                ema_slow=indicator_cfg.get('ema_long', 21),
            )
            if feats.empty:
                continue

            X_list.append(feats.iloc[-1])
            y_list.append(1 if row['result'].lower() == 'win' else 0)
        except Exception as e:
            logger.warning(f"Erro ao processar linha do log: {e}")

    if not X_list:
        logger.error("Nenhum dado válido para treinar o log model.")
        return

    X = pd.DataFrame(X_list)
    y = np.array(y_list)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"XGB Log Model ROC AUC: {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")
    print(f"XGB Log Model Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

    model.fit(X, y)
    joblib.dump(model, model_output)
    print(f"Log model saved to {model_output}")


if __name__ == "__main__":
    # Exemplo de uso:
    train_entry_model()
    # train_log_model()
