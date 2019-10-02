from ..include.dataloader import DataLoader


def test_dataloader_0():
    return
    dataloader = DataLoader()
    print(dataloader.submission[1])
    print(dataloader.queries[10])
    for raw_html in dataloader.raw_content():
        print(raw_html)
        break


def test_dataloader_1():
    dataloader = DataLoader()
    for data in dataloader.processed_content(10):
        print(data)
        break
