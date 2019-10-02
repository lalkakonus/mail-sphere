import sys
from collections import Counter
from dataloader import DataLoader

class DataProcesser():
    
    def __init__(self):
        dl = DataLoader()
        self.idf = None

    def CreateIDF():
        data_dir = os.path.join(dl.data_dir, 'tmp')
        data_list = so.listdir(data_dir)

        idf = Counter()
        for counter, filepath in enumerate(data_list, 1):
            logger.info('Part {} loading'.format(counter))
            for word_list in serializer.load(filepath).values()
                idf.update(word_list)
        
        serializer.save(idf, os.path.join(data_dir, 'idf.dat'))
    
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

    def load_preprocessed_data():
        data = {}
        for counter, filepath in enumerate(os.listdir('../data/tmp'), 1):
            logger.info('Part {} loading'.format(counter))
            update = serializer.load('../data/tmp/' + filepath)
            if not update:
                logger.error('Part {} loading error. Tereminate process'.format(counter))
                break
            #data.update(update)
            logger.info('Part {} loaded'.format(counter))
        return data


# preprocess_data()
    data = load_preprocessed_data()

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
