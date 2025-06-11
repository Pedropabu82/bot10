"""Entry point for starting the trading bot and selecting its mode."""

import asyncio
import json
import logging
import os
from api_client import BinanceClient
from live_strategy import LiveMAStrategy
from backtest_engine import simulate_trades
from websocket_client import start_streams

# Configurar logging para console e arquivo
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

async def main():
    ws_task = None
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting bot...")
        with open('config.json', 'r') as f:
            cfg = json.load(f)
        cfg['api_key'] = os.environ.get('BINANCE_API_KEY', cfg.get('api_key'))
        cfg['api_secret'] = os.environ.get('BINANCE_API_SECRET', cfg.get('api_secret'))
        logger.debug(f"Config loaded: {cfg}")
        client = BinanceClient(cfg)
        strategy = LiveMAStrategy(client, cfg)
        logger.info("Initializing strategy...")
        await strategy.initialize()

        mode = cfg.get('mode', 'live')
        if mode == 'train':
            logger.info("Running in training mode...")
            await strategy.train_mode(cfg.get('train_days', 30))
        elif mode == 'backtest':
            logger.info("Running in backtest mode...")
            metrics, equity = simulate_trades()
            logger.info(f"Backtest results: {metrics}")
        else:
            logger.info("Running in live mode...")
            ws_task = asyncio.create_task(
                start_streams(
                    strategy.symbols,
                    strategy.timeframes,
                    strategy,
                    strategy.timeframes,
                    cfg,
                )
            )
            await strategy.run()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        if ws_task:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass
        if 'client' in locals():
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())
