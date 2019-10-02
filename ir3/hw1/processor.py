# coding: utf-8
from include.idf import IDF 
from include.dataprocessor import RawDataProcessor

class Processor():

    def __init__(self):
        self._idf = IDF()
        self._raw_data_processor = RawDataProcessor()
        
    def run(self, create_idf=False, process_raw_data=False):
        if create_idf:
            self._idf.create()
        if process_raw_data:
            self._raw_data_processor.run()

if __name__ == '__main__': 
    dataprocessor = Processor()
    dataprocessor.run(process_raw_data=True, create_idf=False)
