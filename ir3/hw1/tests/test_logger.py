from ..include.logger import logger
from logging import FileHandler
import logging

def test_logger():
    print(__name__)
    root_logger = logging.getLogger("")
    # hd = logger.handlers[1]
    # logger.removeHandler(hd)
    logger.addHandler(FileHandler("logs/test_tmp.txt", mode="w"))
    print(root_logger.handlers)
    print(logger.hasHandlers())
    logger.info("test test")
