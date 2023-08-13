# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

class testage:
    def testing_model(self,y_test,y_pred,X_train):
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        accuracy_score(y_test, y_pred)

        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
        print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
        print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))