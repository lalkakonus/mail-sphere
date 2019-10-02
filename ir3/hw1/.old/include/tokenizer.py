import pymorphy2
from cachetools import cached, LRUCache
MORPH = pymorphy2.MorphAnalyzer()

def convert_to_lower(function):
    def convert(text):        
        return function(text.lower())
    return convert

def easy_tokenizer(text):
    word = ''
    for symbol in text:
        if symbol.isalnum():
            word += symbol
        elif word:
            yield word
            word = ''

cache = LRUCache(maxsize=10000)

@convert_to_lower
@cached(cache)
def pymorphy_tokenizer(text):
    global PYMORPHY_CACHE
    for word in easy_tokenizer(text):
            # TODO decide weather to save word in norma form or not   
            # word = MORPH.parse(word)[0].normal_form
        yield word
