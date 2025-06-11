"""Evaluate indicator data or a trained model to generate trading signals."""

import joblib
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class SignalEngine:
    def __init__(self, model_path="model_xgb.pkl"):
        self.model = None
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("SignalEngine: Modelo XGBoost carregado com sucesso.")
            else:
                logger.warning("SignalEngine: Modelo não encontrado, fallback ativado.")
        except Exception as e:
            logger.warning(f"SignalEngine: Falha ao carregar modelo ({e}); fallback ativado.")

    def get_signal_for_timeframe(self, data, parameters=None, symbol=None, timeframe=None):
        """Return binary signal using ensemble of indicators or the trained model."""

        parameters = parameters or {}
        try:
            weights = parameters.get("signal_weights", {})

            # sinais individuais
            macd_hist_pos = 1 if data.get("macd_hist", 0) > 0 else 0
            adx_trend = 1 if data.get("adx", 0) > parameters.get("adx_thresh", 25) else 0
            rsi = data.get("rsi", 50)
            rsi_sig = 1 if rsi < parameters.get("rsi_oversold", 30) else 0
            ema_cross = 1 if data.get("ema_fast", 0) > data.get("ema_slow", 0) else 0

            ensemble_score = (
                ema_cross * weights.get("ema", 0.25)
                + rsi_sig * weights.get("rsi", 0.25)
                + macd_hist_pos * weights.get("macd", 0.25)
                + adx_trend * weights.get("adx", 0.25)
            )

            proba = ensemble_score
            ok = ensemble_score > parameters.get("entry_thresh", 0.5)
            decision = "✅" if ok else "❌"

            if self.model:
                features = np.array([[v for v in data.values()]], dtype=float)
                proba = float(self.model.predict_proba(features)[0][1])
                ok = proba >= parameters.get("model_thresh", 0.5)
                decision = "✅" if ok else "❌"

            logger.info(
                f"[AI] {symbol or ''} {timeframe or ''} - proba: {proba:.4f} - decision: {decision}"
            )
            return {"ok": ok, "confidence": round(proba, 4)}

        except Exception as e:
            logger.error(f"Erro ao gerar sinal: {e}")
            return {"ok": False, "confidence": 0.0}
