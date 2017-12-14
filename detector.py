# detector.py, detecting credit card fraud using machine learning
# For the Bank of America codeweek fair 2017
# Mentor: Tufano, Deanna

import csv
import time

import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def visualize_graph(clf, header, output):
    # visualize graph
    dot_data = StringIO()
    tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=header,
                         filled=True, rounded=True,
                         impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(output)


def get_data(file):
    print("getting data....")
    with open(file, 'rt') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
        header = data[0][0:29]
        labels = list()
        features = list()

        print("formatting and converting data...")

        # get features and labels
        count = 0
        iterdata = iter(data)
        next(iterdata)  # skip the headers
        for index, value in enumerate(iterdata):
            labels.append(bool(int(value[29])))  # convert 0 to false and 1 to true
            features.append(
                [float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list
            count += 1

    return header, labels, features, count


def split_data(features, labels, count, ratio):
    # x = features, y = targets
    print("Splitting training & test data...")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=ratio)
    print("There were ", y_train.count(True), " fraud occurrences in ", int(count * ratio), " training samples.")

    return x_train, x_test, y_train, y_test


def train_clf(clf, x_train, y_train):
    print("Training classifier...")
    begin_time = time.time()
    clf.fit(x_train, y_train)
    print("Training done! Training duration: ", time.time() - begin_time, " seconds.")

    return clf


def metrics(clf, x_test, y_test):
    print("Calculating accuracy metrics:")
    predictions = clf.predict(x_test)
    score = accuracy_score(y_test, predictions)

    return score


# # training KNN
# print("Training KNN...")
# begin_time = time.time()
# knn_clf = KNeighborsClassifier()
# knn_clf.fit(feature_train, target_train)
# print("Decision tree training done! Training duration: ", time.time() - begin_time, " seconds.")
#
# # knn metrics
# print("Calculating KNN accuracy metrics:")
# knn_predictions = knn_clf.predict(feature_test)
# knn_score = accuracy_score(target_test)
# print("KNN accuracy: ", knn_score)


def test_datum(clf, datum):
    x = input("Enter test datum (q to quit): ")

    datum = list(map(float, datum.split('\t')))

    return clf.predict(datum), clf.predict_proba(datum)


def main():
    print("Welcome.")

    file = input("Enter data file: ")
    headers, labels, features, count = get_data(file)

    ratio = input("Enter split ration: ")
    x_train, x_test, y_train, y_test = split_data(features, labels, count, ratio)

    knn_clf = KNeighborsClassifier()
    tree_clf = tree.DecisionTreeClassifier()

    knn_clf = train_clf(knn_clf, x_train, y_train)
    tree_clf = train_clf(tree_clf, x_train, y_train)

    knn_score = metrics(knn_clf, x_test, y_test)
    tree_score = metrics(tree_clf, x_test, y_test)
    print("Tree classifier accuracy score: ", tree_score, " KNN Classifier accuracy score: ", knn_score)


if __name__ == "__main__":
    main()
