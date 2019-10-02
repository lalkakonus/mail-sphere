import sys, os
from collections import Counter
import logging.config, logging
from dataloader import DataLoader
import serializer

logging.config.fileConfig(fname='logging.conf', disable_existing_loggers=True)
logger = logging.getLogger('logger.DataProcessor')

class DataProcessor():
    
    def __init__(self):
        self.dl = DataLoader()
        self._idf = None

    @property
    def idf(self):
        if self._idf is None:
            data_dir = os.path.join(os.path.dirname(__file__), '../data/')
            idf_filepath = os.path.join(data_dir, 'idf.dat')
            logger.info('Loading IDF from {}'.format(idf_filepath))
            self._idf = serializer.load(idf_filepath)
            
        if self._idf is not None:
            return self._idf
        else:
            raise ValueError('IDF is not define')

    def CreateIDF(self):
        logger.info('Creating IDF')
        data_dir = os.path.join(os.path.dirname(__file__), '../data/')
        data_dir = os.path.join(data_dir, 'tmp')
        data_list = os.listdir(data_dir)
        if not data_list:
            raise Exception('Data not preprocessed')

        self._idf = Counter()
        for cnt, filepath in enumerate(data_list, 1):
            logger.info('Part {} processing'.format(cnt))
            for word_list in serializer.load(os.path.join(data_dir, filepath)).values():
                self._idf.update(word_list)
        
        idf_filepath = os.path.join(data_dir, 'idf.dat')
        
        if os.path.isfile(idf_filepath):
            ans = input('file {} alredy exist, replace it? (Y/N)'.format(idf_filepth))
            if ans.lower() != 'y':
                return
                
        serializer.save(self._idf, idf_filepath)
    
    def preprocess_data(self):
        BATCH_SIZE = 7000
        data = {}
        counter = 0
        for counter, item in enumerate(self.dl.content(), 1):
            data[item[0]] = item[1]
            if counter  % BATCH_SIZE == 0:
                filename = '../data/tmp/' + str(counter // BATCH_SIZE) + '.dat'
                if not serializer.save(data, filename):
                    return False
                data = {}
        if counter % BATCH_SIZE != 0:
            filename = '../data/tmp/' + str(counter // BATCH_SIZE + 1) + '.dat'
            serializer.save(data, filename)
        logger.info('Data handle finished. {} files processed'.format(counter))
        return True

    def load_preprocessed_data(self):
        data = {}
        for counter, filepath in enumerate(os.listdir('../data/tmp'), 1):
            logger.info('Part {} loading'.format(counter))
            update = serializer.load('../data/tmp/' + filepath)
            if not update:
                logger.error('Part {} loading error. Tereminate process'.format(counter))
                break
            for key, value in update.items():
                serializer.save(value, '../data/processed/' + str(key) + '.dat')
            logger.info('Part {} loaded'.format(counter))
        return data

    def tf(query, doc_id):
        doc = serializer.load('../data/processed/' + str(doc_id) + '.dat')
        tf = np.zeros(len(query))
        for i, word in enumerate(query):
            tf[i] = doc.count(word)
        tf /= len(doc)
        return tf


dp = DataProcessor()
dp.load_preprocessed_data()
'''
print(dl.urls[1])   
print(dl.queries[1])
print(dl.submission[1])
t = time()
for i in dl.content():
    print(i[0])

t = time() - t
print(t / 100 * 75e3 / 60)
'''
