from ..include.dataprocessor import RawDataProcessor
from ..include.dataloader import DataLoader


def test_dataprocessor():
    return
    dataloader = DataLoader()
    dataprocessor = RawDataProcessor()
    for raw_html in dataloader.raw_content():
        data = dataprocessor.mangling(**raw_html)
        print(data)
        break

def test_dataprocessor_1():
    dataloader = DataLoader()
    dataprocessor = RawDataProcessor()
    html = "123.html\n" + open("tests/1.html", "r").read()
    print(dataprocessor.mangling("a", 0, html))

def test_dataprocessor_multiprocessing():
    return
    dataprocessor = RawDataProcessor()
    dataprocessor.run()
                
def test_max_size_processing():
    return
    dataprocessor = RawDataProcessor()
    with open("data/raw/doc.68884.dat", "r") as raw_file:
        url = raw_file.readline()[:-1]
        url_id = 1
        html_data = raw_file.read()
        raw_html = {
            "url": url,
            "url_id": url_id,
            "html_data": html_data}
        data = dataprocessor.mangling(**raw_html)
