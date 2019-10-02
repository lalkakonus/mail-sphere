import sys
import os
sys.path.insert(0, 'include')
from classifier import Classifier
import pandas as pd
import numpy as np
import time

def make_prediction(dataset_filepath, submission_filepath):
    print '# Make prediction for data:', dataset_filepath
    dataset = pd.read_csv(dataset_filepath, header=0, index_col=0)
    print('  * Dataset loaded')
    
    model = Classifier()
    model.load()
    print '  * Model loaded from', model.filepath
    #X = dataset.iloc[:, 1:-1]
    X = dataset.iloc[:, :-1]
    X = model.scaler.transform(X)

    print '  * Making prediction ...'
    prediction = np.zeros((X.shape[0], 2), dtype=np.int64)
    # prediction[:, 0] = dataset.iloc[:, 0]
    prediction[:, 0] = dataset.index
    t1 = time.time()
    prediction[:, 1] = model.predict(X)
    t2 = time.time()
    print '  * Prediction finished (' + str(t2 - t1) + ' sec)'
    
    df = pd.DataFrame(prediction, columns=['Id', 'Prediction']).set_index('Id')
    df.to_csv(submission_filepath)
    print('  * Submission saved to: ' + submission_filepath)
    
if __name__ == '__main__':
    # dataset_filepath = 'data/dataset_test_5000.csv'
    dataset_filepath = 'data/test_data.csv'
    submission_filepath = 'data/my_submission.csv'
    if os.path.isfile(dataset_filepath) == False:
        print 'Error, dataset not found'
        sys.exit()
    make_prediction(dataset_filepath, submission_filepath)
