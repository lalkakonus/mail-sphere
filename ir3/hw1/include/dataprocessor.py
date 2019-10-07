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
from bs4.element import Comment
logger = get_logger(__name__)


class RawDataProcessor():
    
    def __init__(self):
        self._dataloader = DataLoader()
        self._idf = None
        self.tokenizer = Tokenizer()
        self.workers_number = 4 
        self.serializer = Serializer()
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        # ATTENTION !!!
        for filename in os.scandir(self._settings["directory"]["processed_content"]):
            os.remove(os.path.join(filename))

    def find_tag_text(self, bs, tags):
        bs_tags = bs.find_all(tags)
        name = "<" + ">;<".join(tags) + ">"
        result = []
        if len(bs_tags):
            tag_string_array = [tag.string for tag in bs_tags if tag]
            result = [self.tokenizer(text) for text in tag_string_array if text]
            result = [string for string in result if string]
        return {name: result}

    def mangling(self, url, url_id, html_data):
        
        def tag_visible(element):
            if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                return False
            if isinstance(element, Comment):
                return False
            return True
        
        data = {
            "url": url,
            "url_id": url_id,
            "body": None}

        soup = BeautifulSoup(html_data, self._settings["bs"]["parser_type"])
      
        for tags in self._settings["tags"]:
            data.update(self.find_tag_text(soup, tags))

        if self._settings["boilerpipe"]["activate"]:
            extractor = Exctractor(self._settings["boilerpipe"]["type"], html_data)
            text = extractor.getText()
        else:
            texts = soup.findAll(text=True)
            visible_texts = filter(tag_visible, texts)  
            text = u" ".join(t.strip() for t in visible_texts)        
        data["body"] = self.tokenizer(text)
        return data

    def interface(self, queue, worker_id):
        try:
            stop = self._dataloader.raw_content_len
            for raw_data in self._dataloader.raw_content(start=worker_id, stop=stop, step=self.workers_number):
                try:
                    data = self.mangling(**raw_data)
                    filename = "{}.data".format(data["url_id"])
                    self.serializer.save(data, os.path.join(self._settings["directory"]["processed_content"], filename))
                    queue.put(1)
                except Exception as error:
                    logger.error('Error in thread {} occured: {}'.format(os.getpid(), error), exc_info=True)
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
            N = self._dataloader.raw_content_len
            bar = progressbar(range(N))
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
