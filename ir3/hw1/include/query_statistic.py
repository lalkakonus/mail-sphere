from progressbar import progressbar
from .serializer import Serializer
from .dataloader import DataLoader 
from .logger import get_logger
from .idf import IDF
from . import PROCESSING_CONFIG_FILEPATH
import json
logger = get_logger(__name__)

class Statistic:

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self._dataloader = DataLoader()
        self._data = dict()
    
    def create(self):
        logger.info("Statistic calculation started.")
        bar = progressbar(range(len(self._dataloader.queries)))
        for query_id, doc_ids in self._dataloader.submission.items():
            query = set(self._dataloader.queries[query_id])
            self._data[query_id] = dict.fromkeys(query, 0)
            for doc_id in doc_ids:
                doc = self._dataloader.get_processed_file(doc_id)
                body = set(doc["body"])
                title = set(doc["<title>"][0]) if doc["<title>"] else set()
                doc_words = body | title
                keys = query & doc_words
                for key in keys:
                    self._data[query_id][key] += 1
            bar.__next__()
        logger.info("Statistic calculation finished.")
        return self

    def __call__(self, query_id):
        return self._data[query_id]

    def save(self):
        Serializer.save(self._data, self._settings["filepath"]["statistic"])
        return self

    def load(self):
        self._data = Serializer.load(self._settings["filepath"]["statistic"])
        return self
