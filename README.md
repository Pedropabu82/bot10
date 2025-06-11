# Trading Bot

## Setup and Usage

```bash
mkdir -p data models
pip install -r requirements.txt

# Live trading
python live_strategy.py

# Build consolidated datasets
python data_ingest.py

# Retrain models and optimize weights
python auto_retrain.py

# Train entry signal model
python train_model.py

# Validation utilities (walk-forward / Monte Carlo)
python validation.py
```
