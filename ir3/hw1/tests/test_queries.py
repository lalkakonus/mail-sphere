from ..include.dataloader import DataLoader
from collections import Counter


def test_idf():
    dataloader = DataLoader()
    words = Counter()
    for query in dataloader.queries.values():
        words.update(query)

    print(words.most_common(40))
