import sys

from loguru import logger as l

l.remove()
l.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>",
)
