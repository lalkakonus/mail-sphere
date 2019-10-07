from . import PROCESSING_CONFIG_FILEPATH, MODEL_CONFIG_FILEPATH
from .dataloader import DataLoader 
from .logger import get_logger
from .idf import IDF
import json
import math
from collections import Counter
from functools import reduce
logger = get_logger(__name__)

class Score:

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        with open(MODEL_CONFIG_FILEPATH, "r") as config_file:
            self._settings.update(json.load(config_file))
        self._dataloader = DataLoader()
        self._method = "self." + self._settings["model"]
        self._idf = IDF().load()
        
        self._args = self._settings["parametrs"][self._settings["model"]]["active"]
        self._call = eval(self._method)

    def __call__(self, query_id, doc_id):
        return self._call(query_id, doc_id, **self._args)

    def zero_all(self, query_id, doc_id):
        return 0

    def tf_idf(self, query_id, doc_id, tf_type, idf_type):

        def tf(term, doc, doc_len):
           return doc[term] / doc_len
        
        def log_norm(term, doc, doc_len):
            return 1 + math.log(doc[term] / doc_len)

        tf = eval(tf_type)
        idf = self._idf.chmode(idf_type)

        query = self._dataloader.queries[query_id]
        doc = self._dataloader.get_processed_file(doc_id)
        body_counter = Counter(doc["body"])
        score = 0
        for term in query:
            if term in doc["body"]:
                score += tf(term, body_counter, len(doc["body"])) * idf[term]
        return score

    def bm25f(self, query_id, doc_id, k_2, k_1, zones):
        
        def _tf(term, doc, doc_len):
           return doc[term] / doc_len
        
        def tf1(term, doc, doc_len, avgdl):
            f = _tf(term, doc, doc_len)
            # return f * (k_1 + 1) / (f + k_1 * (1 - b + b * doc_len / avgdl))
            return f / (f + k_1 + doc_len / k_2)

        def tf2(Hdr):
            return Hdr / (1 + Hdr)

        def score_tag(doc, query, tag, doc_len, weight):
            if not doc[tag]:
                return 0
            # if tag == "<a>":
            #     print(doc[tag])
            tag_text = []
            for text in doc[tag]:
                tag_text += text
            tag_counter = Counter(tag_text)
            tag_score = 0
            for term in query:
                if term in tag_counter:
                    tag_score += (tf1(term, tag_counter, doc_len, avgdl) + 0.2 * tf2(weight)) * idf[term]
            return tag_score

        idf = self._idf.chmode("bm25f")
        avgdl = self._dataloader.avgdl
        query = self._dataloader.queries[query_id]
        doc = self._dataloader.get_processed_file(doc_id)
        doc_len = len(doc["body"]) #+ len(doc["<title>"])
        # doc["title"] = [doc["title"]]
        doc["body"] = [doc["body"]]
        if not doc_len:
            return 0

        score = 0
        for zone in zones.values():
            # print(zone)
            score += sum(score_tag(doc, query, tag, doc_len, zone["weight"]) for tag in zone["tags"])
        # print(score)
        return score
