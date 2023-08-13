from scripts.data_preprocessing import data_set_preprocessing
from scripts.cat_boost_model import trainingclass
from scripts.testing import testage
from catboost import CatBoostClassifier

DATASET_PATH = '/home/stanley/Documents/confidential/proj_to_upload/public_projects/part 10 gradient boosting + kfold ross/cat_vs_XG/Data.csv'

with trainingclass() as cat:
    X_train, X_test, y_train, y_test=cat.create_splits(DATASET_PATH)
    cat.cat_training(X_train,y_train)
    y_pred = cat.ypred_return()
    cat.testing_model(y_test,y_pred,X_train)
