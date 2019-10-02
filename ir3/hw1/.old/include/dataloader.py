# coding: utf-8
import sys, os
from bs4 import BeautifulSoup 
from multiprocessing import Process, Queue
import logging.config, logging
from tokenizer import pymorphy_tokenizer

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=True)
logger = logging.getLogger('logger')

class DataLoader():

    def __init__(self, tokenizer=pymorphy_tokenizer):
        self.tokenizer = tokenizer
        data_dir = os.path.join(os.path.dirname(__file__), '../data/')

        # Prepare list of content files
        local_content_dir = 'content/20190128'
        self.content_dir = os.path.join(data_dir, local_content_dir)
        self.content_path_list = os.listdir(self.content_dir)
        self.content_path_list.sort()

        # Process queries data
        queries_filename = 'queries.numerate.txt'
        queries_path = os.path.join(data_dir, queries_filename)
        queries = open(queries_path, 'r').read().split('\n')
        generator = map(lambda x: x.split('\t'), queries)
        self._queries = dict()
        for item in generator:
            if item != ['']:
                self._queries[int(item[0])] = item[1]

        # Process submission data
        submission_filename = 'sample_submission.txt'  
        submission_path = os.path.join(data_dir, submission_filename)
        submission = open(submission_path ,'r').read().split('\n')[1:]
        generator = map(lambda x: x.split(','), submission)
        self._submission = dict()
        for item in generator:
            if item != ['']:
                if int(item[0]) not in self._submission:
                    self._submission[int(item[0])] = {int(item[1])}
                else:
                    self._submission[int(item[0])].add(int(item[1]))

        # Process URL data
        urls_filename = 'urls.numerate.txt'  
        urls_path = os.path.join(data_dir, urls_filename)
        urls = open(urls_path, 'r').read().split('\n')
        generator = map(lambda x: x.split('\t'), urls)
        self._urls = dict()
        for item in generator:
            if item != ['']:
                self._urls[int(item[0])] = item[1]


    @property
    def urls(self):
        return self._urls

    @property
    def submission(self):
        return self._submission

    @property
    def queries(self):
        return self._queries


    def tokenize(self, html_data):
        soup = BeautifulSoup(html_data, 'html.parser')
        
        extracted_tag = ['[document]', 'head', 'title', 'style', 'script']
        for s in soup(extracted_tag):
            s.extract() 
        return list(self.tokenizer(soup.get_text()))
    
    def parse_interface(self, queue, start, step):
        try:
            N = 1000000 # Debug const
            for filename in self.content_path_list[start:N:step]:
                f = open(os.path.join(self.content_dir, filename), 'r')
                html_data = f.read()
                f.close()
                queue.put([int(filename.split('.')[1]), self.tokenize(html_data)])
        except Exception:
            logger.error('Error occured, thread {} stopped'.format(os.getpid()), exc_info=True)
        finally:
            queue.put(None) 

    def content(self):
        processes = []
        results_queue = Queue()    
        WORKERS_NUMBER = 4
        try:
            for i in range(WORKERS_NUMBER):
                processes.append(Process(target=self.parse_interface, 
                                         args=(results_queue, i, WORKERS_NUMBER)))
                processes[-1].start()
                logger.info('Process #{}; pid {} started'.format(i, processes[-1].pid))
            
            complete_workers = 0
            while complete_workers != WORKERS_NUMBER:
                word_list = results_queue.get()
                if word_list is None:
                    complete_workers += 1
                else:
                    yield word_list
        except Exception as e:
           logger.error('Error occured!')
        finally:
            for process in processes:
                logger.debug('Process with pid {} joined'.format(process.pid))
                process.join()
