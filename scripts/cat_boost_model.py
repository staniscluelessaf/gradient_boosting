# Training CatBoost on the Training set
from catboost import CatBoostClassifier

class trainingclass(CatBoostClassifier):
    def cat_training(self,X_train,y_train):
        self.classifier = CatBoostClassifier()
        self.classifier.fit(X_train, y_train)       
    def ypred_return(self):
        y_pred = classifier.predict(X_test)
        return y_pred