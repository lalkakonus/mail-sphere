# coding: utf-8
import sys
sys.path.insert(0, 'include')

import mmh3
from doc_to_word import create_generator
from math import log
from serializer import *
from doc_to_word import doc_to_words

class super_dict(dict):
    def __missing__(self, key):
        return 0

class LangaugeModel():
    def __init__(self, N=4):
        self.N = N
        self.eps = 5e-8
        self.struct_3 = super_dict()
        self.struct_4 = super_dict()

    def save(self, filepath='data/langauge_model.pck'):
        print '* Langauge model saved to \"' + filepath + '\"'
        save([self.N, self.eps, self.struct_3, self.struct_4], filepath)

    def load(self, filepath='data/langauge_model.pck'):
        print '* Langauge model loaded from \"' + filepath + '\"'
        self.N, self.eps, self.struct_3, self.struct_4 = load(filepath, False)

    # Debug version
    '''
    def evaluate(self):
        import gzip, base64
        j = 0
        filepath = 'data/kaggle_train_data_tab.csv.gz'
        
        rate = [[], []]

        with gzip.open(filepath) as input_file:
            
            headers = input_file.readline()

            for line in input_file:
                j += 1
                if j > 6600 and j <= 7000:
                    #if j % 10 == 0:
                    #    print j
                    parts = line.strip().split('\t')
                    pageInb64 = parts[3]
                    html_data = base64.b64decode(pageInb64)
                    mark = int(parts[1])
                    text = doc_to_words(html_data, return_set=False, text_type='visible')
                    #title = 'Spam     ' * mark + 'Not spam ' * (1 - mark)
                    #print title, self.predict(list(text))
                    rate[mark].append(self.predict(text))
        
        print sum(rate[0]) / len(rate[0]), sum(rate[1]) / len(rate[1])
    '''

    def fit(self, filepath):
        print '* Langauge model fitting'
        generator = create_generator(filepath, skip_spam=True, return_set=False, STOP_CONDITION=6300)
       
        for doc, _ in generator:
            n_gram_cnt = len(doc) - self.N + 1
            for num in xrange(n_gram_cnt):
                n_gram_3 = mmh3.hash64(''.join(doc[num: num + self.N - 1]).encode('utf-8'))
                n_gram_4 = mmh3.hash64(''.join(doc[num: num + self.N]).encode('utf-8'))
                self.struct_3[n_gram_3] += 1
                self.struct_4[n_gram_4] += 1
            if n_gram_cnt > 0:
                n_gram_3 = mmh3.hash64(''.join(doc[1 - self.N:]).encode('utf-8'))
                self.struct_3[n_gram_3] += 1
        
        summary_cnt_3 = float(sum(self.struct_3.values()))
        summary_cnt_4 = float(sum(self.struct_4.values()))
        
        for key, _ in self.struct_3.iteritems():
            self.struct_3[key] /= summary_cnt_3
        
        for key, _ in self.struct_4.iteritems():
            self.struct_4[key] /= summary_cnt_4
        
        self.eps = min(self.struct_3.values() + self.struct_4.values()) * 5e-1 
        print '* Langauge model seccessefully fitted'

    def predict(self, doc):
        IndepLH = 0
        n_gram_cnt = len(doc) - self.N + 1
        for num in xrange(n_gram_cnt):
            n_gram_4 = mmh3.hash64(''.join(doc[num: num + self.N]).encode('utf-8'))
            n_gram_3 = mmh3.hash64(''.join(doc[num: num + self.N - 1]).encode('utf-8'))
            probability_A = self.struct_4.get(n_gram_4, 0) 
            probability_B = self.struct_3.get(n_gram_3, 0)
            if probability_A == 0 or probability_B == 0:
                probability = self.eps
            else:
                probability = probability_A / probability_B
            IndepLH += log(probability)
    
        IndepLH /= -n_gram_cnt
        
        return IndepLH
