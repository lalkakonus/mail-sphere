# coding: utf-8
TRACE_NUM = 10
import logging, os
reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%H:%M:%S')

def trace(items_num, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d" % items_num)
        
def trace_worker(items_num, worker_id, trace_num=TRACE_NUM):
    if items_num % trace_num == 0: logging.info("Complete items %05d in worker_id %d" % (items_num, worker_id))

class Output:
    def __init__(self, max_count, trace_num=100):
        self.max_count = 0
        self.trace_num = trace_num
        self.max_count = max_count

    def trace(self, iteration, persist=False):
        percent = 'Error' if self.max_count == 0 else '[' + str(100 * iteration / self.max_count) + '%]'
        if iteration % self.trace_num == 0 or persist:
            # os.system('clear')
            logging.info(' Processing ... ' + percent)
