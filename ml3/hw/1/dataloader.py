import pandas as pd

def processing(dataset):
    
    def wrapper(self, filepath):
        X, y = dataset(self, filepath)
        # Processing
        return X, y

    return wrapper

class DataLoader():
    
    def __init__(self, train_fract=0.7, valid_fract=0.1, test_fract=0.2):
        #assert train_fract + valid_gract + test_gract == 1

        self.train_fract = train_fract
        self.test_fract = test_fract
        self.valid_fract = valid_fract

    @processing
    def __load_dataset(self, filepath: str,):
        """
        Load tsv dataset from filepath

        Parametrs
        --------
        filepath : str

        Return
        -------
        X, y : np.array of shape [n_samples, n_features] and [n_samples] respectively
        """
        
        df = pd.read_csv(filepath, sep=" ")
        X, y = df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy()
        return X, y

    def load(self):
        """
        Load dataset from data directory

        Return
        ------
        X_train, X_test, y_train, y_test : np.array of shape [n_samples_train, n_features],
            [n_samples_test ,n_features], [n_samples_train] and [n_samples_test] respectevely
        """
        train_filepath = "data/spam.train.txt"
        test_filepath = "data/spam.test.txt"

        X_train, y_train = self.__load_dataset(train_filepath)
        X_test, y_test = self.__load_dataset(test_filepath)
        return X_train, X_test, y_train, y_test
