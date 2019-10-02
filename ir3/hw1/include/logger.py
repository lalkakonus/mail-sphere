import logging
import logging.config
from . import LOGGING_CONFIG_FILEPATH

logging.config.fileConfig(fname=LOGGING_CONFIG_FILEPATH, disable_existing_loggers=False)

def get_logger(name):
    return logging.getLogger(str(name))
