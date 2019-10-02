from ..include.idf import IDF


def test_idf():
    idf = IDF().create()
    loaded_idf = idf.load()
    assert idf == loaded_idf
    print(idf["в"])
