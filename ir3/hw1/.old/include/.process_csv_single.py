# coding: utf-8
import sys
sys.path.insert(0, 'include')
from logger import trace_worker
import base64, csv, gzip, zlib
from collections import namedtuple

DocItem = namedtuple('DocItem', ['doc_id', 'is_spam', 'url', 'features'])
TEST_DATA_CNT = 0

if TEST_DATA_CNT < 0:
    TEST_DATA_CNT = 1e5

def load_csv_single(input_file_name, calc_features_f, mode='train'):    
    result = []
    # TEST_DATA_CNT = 10
    # pred = lambda x: x > TEST_DATA_CNT
    
    with gzip.open(input_file_name) if input_file_name.endswith('gz') else open(input_file_name)  as input_file:  
        headers = input_file.readline()
        
        for i, line in enumerate(input_file):
            # if pred(i):
            #    continue
            trace_worker(i, 0)
            parts = line.strip().split('\t')
            url_id = long(parts[0])                                        
            mark = bool(parts[1])                   
            url = parts[2]
            pageInb64 = parts[3]
            html_data = base64.b64decode(pageInb64)
            features = calc_features_f(url, html_data)            
            yield DocItem(url_id, mark, url, features)
        trace_worker(i, 0, 1)
