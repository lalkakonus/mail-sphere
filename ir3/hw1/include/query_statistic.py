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
        
        # dict: query_id -> query_text 
        queries = self._dataloader.queries
        # set: words_in_queries
        all_words = set(sum(queries.values(), []))
        
        # dict: url_id -> query_id
        url_to_query = dict()
        for query_id, url_ids in  self._dataloader.submission.items():
            url_to_query.update({url_id: query_id for url_id in url_ids})
            # set default values to _data
            self._data[query_id] = {word: [0 ,0] for word in queries[query_id]}

        logger.info("Preprocessing finished.")

        bar = progressbar(range(self._dataloader.processed_content_len))
        for doc_id in range(1, self._dataloader.processed_content_len + 1):
            doc = self._dataloader.get_processed_file(doc_id)
            query_id = url_to_query[doc_id]
            
            doc_words = set(doc["body"]) | (set(doc["<title>"][0]) if doc["<title>"] else set())
            intersection = all_words & doc_words
            
            for _query_id, query_statistic in  self._data.items():
                pos = 1
                if query_id == _query_id:
                    pos = 0
                for word, value in query_statistic.items():
                    if word in intersection:
                        value[pos] += 1
            
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
