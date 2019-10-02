# coding: utf-8

import sys
sys.path.insert(0, 'include')

import os.path
from process_csv_single import load_csv_single
from serializer import save, load
import feature_extraction
from IDF import CreateIDF
from lang_model import LangaugeModel
from key_words import get_pop_words, get_distinct_words
from classifier import Classifier
import pandas as pd
import numpy as np

def CreateLangaugeModel(filepath):
    model = LangaugeModel()
    model.fit(filepath)
    model.save()

def CreateKeyWords(load_idf=True):
    path_to_idf = 'data/idf.pck'
    path_to_raw_data = 'data/kaggle_train_data_tab.csv.gz'
    path_to_key_words = 'data/key_words.pck'

    if load_idf:
        idf, doc_cnt = load(path_to_idf)
    else:
        idf, doc_cnt = CreateIDF(path_to_raw_data)
        save([idf, doc_cnt], path_to_idf)

    pop_words = get_pop_words(idf, doc_cnt)
    dist_words = get_distinct_words(idf, doc_cnt)
    save([pop_words, dist_words], path_to_key_words)

def GetFeatures(create_features=True, create_lang_model=False, create_idf=False,
                filepath='data/kaggle_train_data_tab.csv.gz',
                processed_dataset_filepath='dataset.csv'):
    
    path_to_train_dataset = 'data/kaggle_train_data_tab.csv.gz'

    if create_lang_model:
        CreateLangaugeModel(path_to_train_dataset)
    
    if create_idf:
        CreateKeyWords()
    
    if create_features:
        feature_extraction.upload_data()
        print('# Feature creation')
        features = list(load_csv_single(filepath, feature_extraction.html_to_features))
        
        description = ['Id']
        description += ['dots count', 'dash count', 'digits count']
        description += ['<a>', 'words in <a>', '<img>', '<b> and <strong>', 
                        'words in <b> and <strong>', '<i> and <em>', 'words in <i> and <em>', 
                        '<meta>', 'words in <title>', 'words in <meta>']
        description += ['text length', 'average word length', 'compressability',
                       'dist words fraction spam [200]', 'dist words fraction spam [200]',
                       'dist words fraction spam [200]', 'dist words fraction spam [200]',
                       'dist words fraction spam [300]', 'dist words fraction spam [300]',
                       'dist words fraction spam [300]', 'dist words fraction spam [300]',
                       'pop words fraction doc [100]', 'pop words fraction all [100]', 
                       'pop words fraction doc [200]', 'pop words fraction all [200]', 
                       'pop words fraction doc [300]', 'pop words fraction all [300]', 
                       'visible content fraction']
        # description += ['langauge model score']
        description += ['Spam']

        data = np.zeros((len(features), len(description)), dtype=np.float64)
        for i, feature in enumerate(features):
            data[i] = [long(feature[0])] + feature[3] + [int(feature[1])]

        dataset = pd.DataFrame(data)
        dataset.columns = description
        dataset.to_csv(processed_dataset_filepath, index=False)
        print '* Dataset saved to', processed_dataset_filepath

if __name__ == '__main__':
    
    load_dump = True
    create_lang_model = False
    create_idf = False
    # Change filepath here
    filepath = 'data/kaggle_test_data_tab.csv.gz'
    processed_dataset_filepath='data/test_data.csv'
    if os.path.isfile(filepath) == False:
        print 'Error, dataset file not found'
        sys.exit()
    GetFeatures(load_dump, create_lang_model, create_idf, filepath, processed_dataset_filepath)
