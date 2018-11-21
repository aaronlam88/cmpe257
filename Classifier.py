import os
import sys
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# get_data_block_start
from get_data import GetData

getData = GetData()
results = open(os.path.basename(__file__)+'.txt', 'w')

accuracy = {}

symbols = getData.getAllSymbols()

for symbol in symbols:
    accuracy[symbol] = []
    features = getData.getSymbolFeaturesWithoutDate(symbol)
    labels = getData.getSymbolCLFLabels(symbol, 4)

    ########################
    # now the real MA work #
    ########################
    # create train and test data set
    X_test, X_train, y_test,  y_train = train_test_split(features, labels, test_size=.5)

    dtree_classifier = tree.DecisionTreeClassifier()
    sgd_classifier = SGDClassifier(loss="log", penalty="elasticnet")
    svm_classifier = svm.SVC()

    # train the classifier
    dtree_classifier.fit(X_train, y_train)
    sgd_classifier.fit(X_train, y_train)
    svm_classifier.fit(X_train, y_train)
    # do prediction
    dtree_predictions = dtree_classifier.predict(X_test)
    sgd_predictions = sgd_classifier.predict(X_test)
    svm_predictions = svm_classifier.predict(X_test)

    accuracy[symbol].append(str(round(accuracy_score(y_test, dtree_predictions)*100, 2))+'%')
    accuracy[symbol].append(str(round(accuracy_score(y_test, sgd_predictions)*100, 2))+'%')
    accuracy[symbol].append(str(round(accuracy_score(y_test, svm_predictions)*100, 2))+'%')

    # print the result
    print("[INFO] %s: %s" % (symbol, ', '.join(accuracy[symbol])), file=sys.stderr)
    print(symbol + ', ' + ', '.join(accuracy[symbol]), file=results)
