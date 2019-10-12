# coding: utf-8
from .dataloader import DataLoader
from .serializer import Serializer
from collections import namedtuple
from collections import Counter
from progressbar import progressbar
import math
import json
from . import PROCESSING_CONFIG_FILEPATH
from .logger import get_logger
logger = get_logger(__name__)


class IDF():

    data_type = namedtuple("IDF", ["df", "cf", "doc_count"])

    def __init__(self):
        self.dataloader = DataLoader()
        self.serializer = Serializer()
        self._data = self.data_type(0, 0, 0)
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self.filepath = self._settings["filepath"]["idf"]
        self._mode = "idf_smoth"

    def load(self):
        self._data = self.data_type(*self.serializer.load(self.filepath))
        logger.info("IDF have been loaded.")
        return self

    def save(self): 
        self.serializer.save(self._data, self.filepath)
        logger.info("IDF have been saved.")
        return self

    def create(self):
        logger.info("IDF creating start")
        df = Counter()
        cf = Counter()
        bar = progressbar(range(self.dataloader.processed_content_len))
        for data in self.dataloader.processed_content():
            counter = Counter(data["body"])
            cf.update(counter)
            df.update(counter.keys())
            bar.__next__()
        self._data = self.data_type(df, cf, self.dataloader.processed_content_len)
        logger.info("IDF creating finished")
        self.save()
        return self

    def df(self, key):
        return self._data.df.get(key, 0)
 
    def idf(self, key):
        N = self._data.doc_count
        df = self._data.df.get(key, 0)
        if df:
            return math.log(N / df)
        return 0

    def prob_idf(self, key):
        N = self._data.doc_count
        df = self._data.df.get(key, 0)
        if df:
            return max(0, math.log((N / df) - 1))
        return 0

    def bm25yandex(self, key):
        N = self._data.doc_count
        cf = self._data.cf.get(key, 0)
        if cf:
            return math.log(1 - math.exp(-1.5 * cf / N))
        return 0

    def smoth_idf(self, key):
        N = self._data.doc_count
        df = self._data.df.get(key, 0)
        if df:
            return max(math.log((N - df + 0.5) / (df + 0.5)), 0)
        return 0

    def icf(self, key):
        TotalLemms = 1844 * 74000
        cf = self._data.cf.get(key, 0)
        if cf:
            return math.log(TotalLemms / cf)
        return 0

    
    @property
    def doc_count(self):
        return self.dataloader.processed_count_len
