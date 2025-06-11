"""Training utilities for building the XGBoost model from trade logs."""

import pandas as pd
import numpy as np
import xgboost as xgb
import talib
import joblib
import logging
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from features import extract_features

logger = logging.getLogger(__name__)




def train_model(log_path='data/trade_log.csv', model_output='model_xgb.pkl', config_file='config.json'):
    try:
        trades = pd.read_csv(log_path)
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        bb_period = cfg.get('bb_period', 20)
        bb_k = cfg.get('bb_k', 2)
        stoch_k_period = cfg.get('stoch_k_period', 14)
        stoch_d_period = cfg.get('stoch_d_period', 3)
        indicator_cfg = cfg.get('indicators', {})
        trades = trades.dropna()
        trades = trades[trades['type'] == 'ENTRY']

        X_list = []
        y_list = []

        for _, row in trades.iterrows():
            try:
                df = pd.DataFrame({
                    'open': [row['open']],
                    'high': [row['high']],
                    'low': [row['low']],
                    'close': [row['close']],
                    'volume': [row['volume']]
                })
                df = pd.concat([df] * 150, ignore_index=True)  # Simular série temporal
                ema_short = indicator_cfg.get(row['symbol'], {}).get('ema_short', 9)
                ema_long = indicator_cfg.get(row['symbol'], {}).get('ema_long', 21)
                feats = extract_features(
                    df,
                    bb_period=bb_period,
                    bb_k=bb_k,
                    stoch_k_period=stoch_k_period,
                    stoch_d_period=stoch_d_period,
                    ema_short=ema_short,
                    ema_long=ema_long,
                )
                if feats.empty:
                    continue
                X_list.append(feats.iloc[-1])
                y_list.append(1 if row['result'].lower() == 'win' else 0)
            except Exception as e:
                logger.warning(f"Erro ao processar linha: {e}")

        if not X_list:
            logger.error("Nenhum dado válido para treinar.")
            return

        X = pd.DataFrame(X_list)
        y = np.array(y_list)

        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        logger.info("--- Validação Cruzada ---")
        logger.info(f"ROC AUC: {roc_scores.mean():.4f} (+/- {roc_scores.std():.4f})")
        logger.info(f"Accuracy: {acc_scores.mean():.4f} (+/- {acc_scores.std():.4f})")

        model.fit(X, y)
        joblib.dump(model, model_output)
        logger.info(f"Modelo treinado e salvo em: {model_output}")

    except Exception as e:
        logger.error(f"Erro ao treinar modelo: {e}")


if __name__ == '__main__':
    train_model()
