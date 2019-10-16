from ..include.idf import IDF
from collections import Counter

def test_idf():
    return
    idf = IDF().create()
    loaded_idf = idf.load()
    assert idf == loaded_idf
    print(idf["в"])


def test_idf_icf():
    idf = IDF()
    loaded_idf = idf.load()
    cf = Counter(loaded_idf._data.cf)
    print(cf.most_common(100))
