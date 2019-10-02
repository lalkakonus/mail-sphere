import sys
sys.path.insert(0, 'include')
from classifier import Classifier
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

def tune_model(dataset_filepath):
    print('# Performing model tuning')
    dataset = pd.read_csv(dataset_filepath, header=0, index_col=0)
    print '  * Train dataset loaded from ', dataset_filepath
    X, y = dataset.values[:, :-1], dataset.values[:, -1]

    model = Classifier()
    model.scaler.fit(X)
    
    X = model.scaler.transform(X)
    N_SPLITS = 10
    kf = KFold(n_splits=N_SPLITS)
    kf.get_n_splits(X)

    score = np.empty(N_SPLITS, dtype=np.float)
    for iteration, [train_idx, test_idx] in enumerate(kf.split(X)):
        model.train(X[train_idx], y[train_idx])
        score[iteration] = model.evaluate(X[test_idx], y[test_idx])
        print '    - Iteration # ', str(iteration) + ', score:', score[iteration]
    print '  * Mean F1 measure: ', score.mean()
    
    print '  * Save model (Y/N)?'
    ans = sys.stdin.readline()
    if ans[0].lower() == 'y':
        model.train(X, y)
        model.save()
    print '# Exit'


if __name__ == '__main__':
    dataset_filepath = 'data/train_data.csv'
    # dataset_filepath = 'data/dataset_train_5000.csv'
    tune_model(dataset_filepath)
