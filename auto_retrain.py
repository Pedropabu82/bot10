"""Retrain the XGBoost model automatically from logged trades."""

import pandas as pd
import numpy as np
import ccxt
import joblib
import xgboost as xgb
import time
import os
import logging
import json
from sklearn.utils.class_weight import compute_class_weight
from features import extract_features
from validation import walk_forward
import optuna

logger = logging.getLogger(__name__)

def fetch_ohlcv(symbol, timeframe, since, limit=300):
    binance = ccxt.binance({
        'enableRateLimit': True,
    })
    binance.options['defaultType'] = 'future'
    try:
        data = binance.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Erro ao buscar candles para {symbol} {timeframe}: {e}")
        return None


def train_from_log(trade_log='data/trade_log.csv', config_file='config.json'):
    if not os.path.exists(trade_log):
        logger.error(f"Arquivo {trade_log} não encontrado.")
        return

    trades = pd.read_csv(trade_log)
    with open(config_file, 'r') as f:
        cfg = json.load(f)
    bb_period = cfg.get('bb_period', 20)
    bb_k = cfg.get('bb_k', 2)
    stoch_k_period = cfg.get('stoch_k_period', 14)
    stoch_d_period = cfg.get('stoch_d_period', 3)
    indicator_cfg = cfg.get('indicators', {})
    trades = trades.dropna()
    trades = trades[trades['type'] == 'ENTRY']
    logger.info(f"Carregando {len(trades)} trades do log.")

    if len(trades) < 10:
        logger.error("❌ ERRO: Menos de 10 trades disponíveis. Adicione mais dados para treino.")
        return

    X, y = [], []

    for _, row in trades.iterrows():
        symbol = row['symbol']
        timeframe = row['timeframe']
        timestamp = pd.to_datetime(row['timestamp'])
        since = int((timestamp - pd.Timedelta(minutes=600)).timestamp() * 1000)

        df = fetch_ohlcv(symbol, timeframe, since)
        if df is None or df.empty:
            logger.warning(f"Dados vazios para {symbol} {timeframe}, pulando...")
            continue

        ema_short = indicator_cfg.get(symbol, {}).get('ema_short', 9)
        ema_long = indicator_cfg.get(symbol, {}).get('ema_long', 21)
        feats = extract_features(
            df,
            bb_window=bb_period,
            bb_std=bb_k,
            stoch_window=stoch_k_period,
            stoch_smooth=stoch_d_period,
            ema_fast=ema_short,
            ema_slow=ema_long,
        )
        if feats.empty:
            logger.warning(f"Features vazias para {symbol} {timeframe}, pulando...")
            continue

        X.append(feats.iloc[-1])
        y.append(1 if row['result'].lower() == 'win' else 0)

        time.sleep(0.1)

    if not X:
        logger.error("Nenhum dado válido coletado para treino.")
        return

    unique_classes = set(y)
    logger.info(f"Classes encontradas no y: {unique_classes}")
    if len(unique_classes) < 2:
        logger.error("❌ ERRO: Apenas uma classe detectada no vetor y. Adicione mais trades de tipos diferentes (win/loss).")
        return

    df_X = pd.DataFrame(X)
    # Compute class weights for balancing
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=np.array(y))
    weight_dict = {0: class_weights[0], 1: class_weights[1]}
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=weight_dict[1]/weight_dict[0])
    model.fit(df_X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/log_xgb_model.pkl")
    logger.info("✅ Modelo treinado e salvo como models/log_xgb_model.pkl")


def optimize_params(trade_log="data/trade_log.csv", output="best_params.json"):
    """Optimize indicator parameters using walk-forward validation."""
    if not os.path.exists(trade_log):
        logger.error(f"Arquivo {trade_log} não encontrado.")
        return
    trades = pd.read_csv(trade_log)
    if trades.empty:
        logger.error("Trade log vazio para otimização.")
        return

    def objective(trial):
        params = {
            "entry_thresh": trial.suggest_float("entry_thresh", 0.5, 0.9),
            "exit_thresh": trial.suggest_float("exit_thresh", 0.1, 0.5),
            "signal_weights": {
                "ema": trial.suggest_float("w_ema", 0, 1),
                "rsi": trial.suggest_float("w_rsi", 0, 1),
                "macd": trial.suggest_float("w_macd", 0, 1),
                "adx": trial.suggest_float("w_adx", 0, 1),
            },
        }
        folds = list(walk_forward(trades, train_size=50, test_size=10, n_splits=3))
        metric = 0
        for _, test in folds:
            metric += test["pnl_pct"].mean()
        return metric / len(folds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    best = study.best_params
    params = {
        "entry_thresh": best.get("entry_thresh"),
        "exit_thresh": best.get("exit_thresh"),
        "signal_weights": {
            "ema": best.get("w_ema"),
            "rsi": best.get("w_rsi"),
            "macd": best.get("w_macd"),
            "adx": best.get("w_adx"),
        },
    }
    with open(output, "w") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Melhores parâmetros salvos em {output}")

if __name__ == "__main__":
    train_from_log()
    optimize_params()
