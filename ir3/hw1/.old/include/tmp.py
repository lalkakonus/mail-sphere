import logging.config, logging
from dataloader import DataLoader
import serializer

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=True)
logger = logging.getLogger('logger.DataProcessor')
