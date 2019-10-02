# coding: utf-8
import sys
# sys.path.insert(0, 'include')
'''
from logger import trace_worker
import base64, csv, gzip, zlib
from collections import namedtuple
from multiprocessing import Process, Queue

DocItem = namedtuple('DocItem', ['doc_id', 'is_spam', 'url', 'features'])
WORKER_NUM = 4
if TEST_DATA_CNT < 0:
    TEST_DATA_CNT = 1e5

def load_csv_worker(input_file_name, calc_features_f, worker_id, res_queue, mode):    
    if mode == 'train':
        pred = lambda x: x > TEST_DATA_CNT
    else:
        pred = lambda x: x < TEST_DATA_CNT or x > TEST_DATA_CNT + 50
    with gzip.open(input_file_name) if input_file_name.endswith('gz') else open(input_file_name)  as input_file:  
        headers = input_file.readline()
        for i, line in enumerate(input_file):
            if pred(i):
                continue
            trace_worker(i, worker_id)
            if i % WORKER_NUM != worker_id: continue
            parts = line.strip().split('\t')
            url_id = int(parts[0])                                        
            mark = bool(int(parts[1]))                    
            url = parts[2]
            pageInb64 = parts[3]
            html_data = base64.b64decode(pageInb64)
            features = calc_features_f(url, html_data)            
            res_queue.put(DocItem(url_id, mark, url, features))
                
        trace_worker(i, worker_id, 1)  
    res_queue.put(None)
        
def load_csv_multiprocess(input_file_name, calc_features_f, mode='train'):
    processes = []
    res_queue = Queue()    
    for i in xrange(WORKER_NUM):
        process = Process(target=load_csv_worker, args=(input_file_name, calc_features_f, i, res_queue, mode))
        processes.append(process)
        process.start()
    
    complete_workers = 0
    while complete_workers != WORKER_NUM:
        item = res_queue.get()
        if item is None:
            complete_workers += 1
        else:
            yield item
        
    for process in processes: process.join()

'''

import os

data_dir = os.path.join(os.path.dirname(__file__), '../data/')

local_content_directory = 'content/20190128'
content_directory = os.path.join(data_dir, local_content_directory)
content_path_list = os.listdir(conent_path)

queries_filename = 'queries.numerate.txt'
queries_path = os.path.join(data_dir, queries_filename)

submission_filename = 'sample_submission.txt'  
submission_path = os.path.join(data_dir, submission_filename)

urls_filename = 'urls.numerate.txt'  
urls_path = os.path.join(data_dir, urls_filename)


print(conent_path_list[:10], queries_path, submission_path, urls_path)
