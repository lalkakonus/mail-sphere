from .dataloader import DataLoader
from .serializer import Serializer
from .tokenizer import Tokenizer
import os
import json
from collections import Counter
from multiprocessing import Process
from multiprocessing import Queue
from bs4 import BeautifulSoup
from progressbar import progressbar
from . import PROCESSING_CONFIG_FILEPATH
from .logger import get_logger
import logging
logger = get_logger(__name__)


class RawDataProcessor():
    
    def __init__(self):
        self._dataloader = DataLoader()
        self._idf = None
        self.tokenizer = Tokenizer()
        self.workers_number = 2
        self.serializer = Serializer()
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        for filename in os.scandir(self._settings["directory"]["processed_content"]):
            os.remove(os.path.join(filename))

    def mangling(self, url, url_id, html_data):
        data = {
            "url": url,
            "url_id": url_id,
            "title": None,
            "links": None,
            "body": None}

        soup = BeautifulSoup(html_data, 'html.parser')
      
        title = soup.title
        if title is not None:
            data["title"] = self.tokenizer(title.string)
        
        links = soup.find_all("a")
        if len(links):
            tmp = list(filter(lambda text: text is not None, [link.string for link in links]))
            data["links"] = list(filter(lambda x: len(x), list(self.tokenizer(text) for text in tmp)))

        extracted_tag = ['[document]', 'head', 'title', 'style', 'script']
        for s in soup(extracted_tag):
             s.extract()
        data["body"] = self.tokenizer(soup.get_text())

        return data

    def interface(self, queue, worker_id):
        try:
            stop = 1000 # Debug const
            for raw_data in self._dataloader.raw_content(start=worker_id, stop=stop, step=self.workers_number):
                data = self.mangling(**raw_data)
                filename = "{}.data".format(data["url_id"])
                self.serializer.save(data, os.path.join(self._settings["directory"]["processed_content"], filename))
                queue.put(1)
        except Exception as error:
            logger.error('Error in thread {} occured: {}'.format(os.getpid(), error), exc_info=True)
        finally:
            queue.put(None)

    def run(self):
        processes = []
        queue = Queue()
        try:
            for worker_id in range(self.workers_number):
                processes.append(Process(target=self.interface, 
                                         args=(queue, worker_id)))
                processes[-1].start()
                logger.info('Process #{}; pid {} started'.format(worker_id, processes[-1].pid))
 
            complete_workers = 0
            progress = 0
            bar = progressbar(range(1000))
            while complete_workers != self.workers_number:
                update = queue.get()
                if update is None:
                    complete_workers += 1
                else:
                    progress += 1
                try:
                    next(bar)
                except StopIteration:
                    pass
            logger.info("All data processed")
        except Exception as error:
            logger.error('Error occured: {}, all processes will be terminated.'.format(error), exc_info=True)
            for process in processes:
                pid = process.pid
                process.terminate()
                logger.info('Process with pid {} terminated'.format(pid))
            return
        finally:
            for process in processes:
                pid = process.pid
                process.join()
                logger.info('Process (pid - {}) joined'.format(pid))
