# coding: utf-8
from .serializer import Serializer
import sys
import os
import json
from .tokenizer import Tokenizer
from . import PROCESSING_CONFIG_FILEPATH
from .logger import get_logger
logger = get_logger(__name__)


class DataLoader():

    def __init__(self):
        with open(PROCESSING_CONFIG_FILEPATH, "r") as config_file:
            self._settings = json.load(config_file)
        
        # Prepare list of raw content files
        raw_content_dir = self._settings["directory"]["raw_content"]
        self._raw_content_list = list(os.scandir(raw_content_dir))
        
        # Prepare list of processed content files
        processed_content_dir = self._settings["directory"]["processed_content"]
        self._processed_content_list = list(os.scandir(processed_content_dir))

        # Process queries data
        tokenizer = Tokenizer()
        queries_filepath = self._settings["filepath"]["queries"]
        with open(queries_filepath, 'r') as queries_file:
            queries = queries_file.read().split('\n')
            generator = map(lambda x: x.split('\t'), queries)
            self._queries = dict()
            for item in generator:
                if item != ['']:
                    self._queries[int(item[0])] = tokenizer(str(item[1]))

        # Process submission data
        submission_filepath = self._settings["filepath"]["sample_submission"]
        with open(submission_filepath, 'r') as submission_file:
            submission = submission_file.read().split('\n')[1:]
            generator = map(lambda x: x.split(','), submission)
            self._submission = dict()
            for item in generator:
                if item != ['']:
                    if int(item[0]) not in self._submission:
                        self._submission[int(item[0])] = {int(item[1])}
                    else:
                        self._submission[int(item[0])].add(int(item[1]))

        # Process URL data
        urls_filepath = self._settings["filepath"]["urls"]
        with open(urls_filepath, 'r') as urls_file:
            urls = urls_file.read().split('\n')
            generator = map(lambda x: x.split('\t'), urls)
            self._urls = dict()
            for item in generator:
                if item != ['']:
                    self._urls[item[1]] = int(item[0])

    def __str__(self):
        return "DataLoader"

    @property
    def urls(self):
        return self._urls

    @property
    def submission(self):
        return self._submission

    @property
    def queries(self):
        return self._queries
    
    def raw_content(self, start=0, stop=-1, step=1):
        for raw_filepath  in self._raw_content_list[start: stop: step]:
            try:
                with open(raw_filepath, "r") as raw_file:
                    url = raw_file.readline()[:-1]
                    url_id = self.urls[url]
                    html_data = raw_file.read()
                    yield {
                        "url": url,
                        "url_id": url_id,
                        "html_data": html_data}
            except Exception as error:
                logger.error("Error occured during raw content parsing file: {}. {}".format(raw_filepath, error), exc_info=True)
                continue

    @property
    def raw_content_len(self):
        return len(self._raw_content_list)

    def processed_content(self, start=0, stop=-1, step=1):
        for filepath in self._processed_content_list[start: stop: step]:
            yield Serializer.load(filepath)
    
    def get_processed_file(self, doc_id):
        filepath = self._settings["directory"]["processed_content"] + "/" + str(doc_id) + ".data"
        assert os.path.isfile(filepath)
        return Serializer.load(filepath)

    @property
    def avgdl(self):
        return 1800
        # return self._avgdl

    @property
    def processed_content_len(self):
        return len(self._processed_content_list)
