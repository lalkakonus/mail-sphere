# coding: utf-8
# from nltk.corpus import stopwords
from pymystem3 import Mystem
# import pymorphy2
import json
from string import punctuation
from . import PROCESSING_CONFIG_FILEPATH
# from cachetools import cached, LRUCache


class Tokenizer():
    
    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        self.stopwords = []
        self.mystem = Mystem()
        # self.pymorph = pymorphy2.MorphAnalyzer()
        self.lemmatizer = lambda word: word

        if self._settings["stop_words"]["remove"]:
            pass
            # self.stopwords = stopwords.words("russian")
        if self._settings["stemming"]["activate"]:
            if self._settings["stemming"]["stemmer"] == "mystem":
                self.lemmatizer = lambda text: [self.mystem.lemmatize(word)[0] for word in text]
            if self._settings["stemming"]["stemmer"] == "pymorhy":
                pass
                # self.lemmatizer = lambda text: [self.pymorph.parse(word)[0].normal_form for word in text]
    
    def tokenizer(self, text):
        text = str(text).lower()
        result = []
        word = ''
        for symbol in text:
            if symbol.isalnum():
                word += symbol
            elif word:
                if word not in self.stopwords:
                    result.append(word)
                word = ''
        if word and word not in self.stopwords:
            result.append(word)
        return result

    def __call__(self, text):
        if not text:
            return []
        return self.lemmatizer(self.tokenizer(text))
