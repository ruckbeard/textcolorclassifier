import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import fbeta_score, accuracy_score
import pickle

# Load the Census dataset
data = pd.read_csv("data.txt", sep=" ", header=None)
data.columns = ["red", "blue", "green", "text_color"]
data["red"] = data.red.str.replace(',', '').astype(int)
data["blue"] = data.blue.str.replace(',', '').astype(int)
data["green"] = data.green.str.replace(',', '').astype(int)
data["text_color"] = data.text_color.astype(int)

features_final = data.drop("text_color", axis=1)

print(features_final.dtypes)

text_color = data["text_color"]

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final,
                                                    text_color,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

TP = np.sum(text_color)
FP = text_color.count() - TP
TN = 0
FN = 0
accuracy = (TP + TN) / (TP + FP + TN + FN)
recall = TP / (TP + FN)
precision = TP / (TP + FP)

fscore = (1.25)*((precision*recall)/((0.25*precision)+recall))

# Print the results
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))

clf = AdaBoostClassifier(random_state = 1)

parameters = {'n_estimators' : [50,75,100,200], 'learning_rate' : [0.5,0.8,1,1.2]}
scorer = make_scorer(fbeta_score, beta=0.5)
grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
grid_fit = grid_obj.fit(X_train, y_train)
best_clf = grid_fit.best_estimator_
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, "wb"))