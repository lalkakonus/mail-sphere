from . import PROCESSING_CONFIG_FILEPATH
from .dataloader import DataLoader 
from .logger import get_logger
from .idf import IDF
import json
from collections import Counter
import math.log
logger = get_logger(__name__)

class Score:

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self._method = "self." + self._settings["score_method"]
        self._dataloader = DataLoader()
        self._idf = IDF().load()
        
        self._args = self._settings["parametrs"][self._method]["active"]
        self._call = eval(self._method)

    def __call__(self, query_id, doc_id):
        return self._call(query_id, doc_id, **self._args)

    def zero_all(self, query_id, doc_id):
        return 0

    def tf_idf(self, query_id, doc_id, tf_type, idf_type):

        def tf(term, doc, doc_len):
           return doc[term] / doc_len
        
        def log_norm(term, doc, doc_len):
            return 1 + math.log(tf_tf(term, doc, doc_len))

        tf = eval(tf_type)
        idf = self._idf.chmode(idf_type)

        query = self._dataloader.queries[query_id]
        doc = self._dataloader.processed_content[doc_id]
        body_counter = Counter(doc["body"])
        score = 0
        for term in query:
           score += tf(term, body_counter, len(doc["body"])) * idf(term)
