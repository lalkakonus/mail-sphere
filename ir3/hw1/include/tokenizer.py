# coding: utf-8
import pymorphy2
# from cachetools import cached, LRUCache

class Tokenizer():
    # MORPH = pymorphy2.MorphAnalyzer()
    # cache = LRUCache(maxsize=10000)
    
    def easy_tokenizer(self, text):
        word = ''
        for symbol in text:
            if symbol.isalnum():
                word += symbol
            elif word:
                yield word
                word = ''
        if len(word):
            yield word


    # @cached(cache)
    def __call__(self, text):
        # global PYMORPHY_CACHE
        text = str(text)
        if not len(text):
            return []
        return list(word for word in self.easy_tokenizer(text.lower()))
        # TODO decide weather to save word in norma form or not   
        # word = MORPH.parse(word)[0].normal_form
        #yield word
