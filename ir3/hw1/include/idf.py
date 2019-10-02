# coding: utf-8
from .dataloader import DataLoader
from .serializer import Serializer
from collections import namedtuple
from collections import Counter
from progressbar import progressbar
import json
from . import PROCESSING_CONFIG_FILEPATH
from .logger import get_logger
logger = get_logger(__name__)


class IDF():

    data_type = namedtuple("IDF", ["word_count", "doc_count"])

    def __init__(self):
        self.dataloader = DataLoader()
        self.serializer = Serializer()
        self._data = self.data_type(None, 0)
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self.filepath = self._settings["filepath"]["idf"]
        self._mode = "idf_smoth"

    def chmode(self, mode):
        self._mode = mode
        return self

    def load(self):
        logger.info("IDF loading")
        self._data = self.data_type(*self.serializer.load(self.filepath))
        return self

    def save(self): 
        logger.info("IDF saving")
        self.serializer.save(self._data, self.filepath)
        return self

    def create(self):
        logger.info("IDF creating start")
        idf = Counter()
        bar = progressbar(range(self.dataloader.processed_content_len))
        for data in self.dataloader.processed_content():
            bar.__next__()
            idf.update(set(data["body"]))
        self._data = self.data_type(idf, self.dataloader.processed_content_len)
        logger.info("IDF creating finished")
        self.save()
        return self
    
    def __getitem__(self, key):
        if self._mode == "idf_smoth":
            return math.log(self._data.doc_count / (1 + self._data.word_count.get(key, 0)))

    @property
    def doc_count(self):
        return self.dataloader.processed_count_len
