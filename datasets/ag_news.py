import numpy as np
import pandas as pd
from torchtext.datasets import AG_NEWS

path = "../data/AG_NEWS/"

class AGNewsDataset:
    def __init__(self, path, ngrams=3):
        self.path = path
        self.train_iter, self.test_iter = AG_NEWS(root = path)#, ngrams=ngrams)
        
    def get_train_data(self):
        return pd.DataFrame(self.train_iter._data, columns=["targets", "data"])

    def get_test_data(self):
        return pd.DataFrame(self.test_iter._data, columns=["targets", "data"])


