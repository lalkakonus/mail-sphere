from ..include.dataprocessor import RawDataProcessor
from ..include.dataloader import DataLoader


def test_dataprocessor():
    dataloader = DataLoader()
    dataprocessor = RawDataProcessor()
    for raw_html in dataloader.raw_content():
        data = dataprocessor.mangling(**raw_html)
        # print(data["body"])
        break

def test_dataprocessor_multiprocessing():
    dataprocessor = RawDataProcessor()
    dataprocessor.run()
