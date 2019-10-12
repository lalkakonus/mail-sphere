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
    return
    dataloader = DataLoader()
    for data in dataloader.processed_content(10):
        print(data)
        break

def test_dataloader_2():
    dataloader = DataLoader()
    # print(dataloader.avgdl)
    # return
    # print(dataloader.queries[1000])
    data = dataloader.get_processed_file(10093)
    print(data)
