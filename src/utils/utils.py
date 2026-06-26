"""
UTILITIES LAYER - Centralized System Logging & Monitoring Infrastructure
"""

import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """
    Creates and configures a centralized system logger.
    """
    # 1. Initialize file handler and define standard structured log format
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 2. Build tracking logger instance capped at the designated execution level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger