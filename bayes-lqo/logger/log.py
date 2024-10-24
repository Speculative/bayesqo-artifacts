import os
import sys

from loguru import logger as logger_init

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger_init.remove()
logger_init.add(
    sys.stdout,
    level=os.getenv("LOG_LEVEL") or "WARNING",
    backtrace=True,
    diagnose=True,
    format="<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)
l = logger_init
