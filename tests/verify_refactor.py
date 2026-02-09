import sys
import os
import pandas as pd
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.signals.signal_generator import SignalGenerator
from config.config import config

def verify_system():
    logger.info("Starting System Verification...")
    
    # 1. Test Feature Engineering (implicitly through SignalGenerator training)
    # We'll use a very short training period and few epochs to verify pipeline works
    config.EPOCHS = 2 # Override for testing
    config.LOOKBACK_PERIOD = 20
    
    symbol = 'AAPL'
    generator = SignalGenerator()
    
    # 2. Test Training
    logger.info(f"Testing Model Training for {symbol}...")
    try:
        # train_model fetches '2y' data by default. 
        # We'll trust it works if it returns True.
        success = generator.train_model(symbol, training_period="6mo")
        if not success:
            logger.error("Training failed.")
            return False
        logger.info("Training successful.")
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. Test Signal Generation
    logger.info(f"Testing Signal Generation for {symbol}...")
    try:
        signal = generator.generate_signal(symbol)
        if signal:
            logger.info(f"Signal Generated: {signal.signal} | Confidence: {signal.confidence:.2f}")
            logger.info(f"Reasoning: {signal.reasoning}")
        else:
            logger.error("Signal generation returned None.")
            return False
    except Exception as e:
        logger.error(f"Signal generation crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    logger.info("Verification Passed!")
    return True

if __name__ == "__main__":
    verify_system()
