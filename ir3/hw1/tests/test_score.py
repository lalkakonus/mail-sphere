from ..include.idf import IDF
from ..include.dataloader import DataLoader
from ..tokeniizer import Tokenizer

def test_idf():
    dataloader = DataLoader()
    for query_num, doc_nums in enumerate(dataloader.submission):
        querey = dataloader.queries[query_num]

    
    sample_submission = dataloader.submission
    idf = IDF().create()
    loaded_idf = idf.load()
    assert idf == loaded_idf
    print(idf["секс"])
