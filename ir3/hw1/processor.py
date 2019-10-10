# coding: utf-8
from include.idf import IDF 
from include.query_statistic import Statistic
from include.dataprocessor import RawDataProcessor

class Processor():

    def __init__(self):
        self._idf = IDF()
        self._raw_data_processor = RawDataProcessor()
        self._statistic = Statistic()
        
    def run(self, create_idf=False, process_raw_data=False, query_statistic=False):
        if create_idf:
            self._idf.create()
        if process_raw_data:
            self._raw_data_processor.run()
        if query_statistic:
            self._statistic.create().save()

if __name__ == '__main__': 
    dataprocessor = Processor()
    dataprocessor.run(process_raw_data=False, create_idf=False, query_statistic=False)
