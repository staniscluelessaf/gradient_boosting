import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scripts.dataset_path as dp
import xgboost
import catboost
from sklearn.model_selection import train_test_split

# Importing the dataset
class data_set_preprocessing:
    def create_splits(self, dataset):
        self.dataset = pd.read_csv(dataset)
        self.X = dataset.iloc[:, :-1].values
        self.y = dataset.iloc[:, -1].values

        # Splitting the dataset into the Training set and Test set
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        return X_train, X_test, y_train, y_test