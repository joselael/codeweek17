# detector.py, detecting credit card fraud using machine learning
# For the Bank of America codeweek fair 2017
# Mentor: Tufano, Deanna

import csv
import time

import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier


# from sklearn


def visualize_graph(clf, header, output):
    dot_data = StringIO()
    tree.export_graphviz(clf,
                         out_file=dot_data,
                         feature_names=header,
                         filled=True, rounded=True,
                         impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(output)


def get_data(file):
    print("getting data...")
    with open(file, 'rt') as csv_file:
        reader = csv.reader(csv_file)
        data = list(reader)
        header = data[0][0:29]
        labels = list()
        features = list()
        fraud_features = list()
        fraud_labels = list()

        print("formatting and converting data...\n")

        # get features and labels
        count = 0
        iterdata = iter(data)
        next(iterdata)  # skip the headers
        for index, value in enumerate(iterdata):

            labels.append(bool(int(value[29])))  # convert 0 to false and 1 to true

            if labels[index]:
                fraud_labels.append(True)
                fraud_features.append([float(i) for i in value[0:29]])

            features.append(
                [float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list
            count += 1

    return header, labels, features, count, fraud_labels, fraud_features


def split_data(features, labels, count, ratio):
    # x = features, y = targets
    print("Splitting training & test data...")
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=ratio)
    print("There were ", y_train.count(True), " fraud occurrences in ", int(count * ratio), " training samples.\n")

    return x_train, x_test, y_train, y_test


def train_clf(clf, x_train, y_train):
    print("Training classifier ", clf.__str__()[:16], "...")
    begin_time = time.time()
    clf.fit(x_train, y_train)
    print("Training done! Training duration: ", time.time() - begin_time, " seconds.\n")

    return clf


# def accuracy_metrics(clf, x_test, y_test):
#     print("Calculating accuracy metrics for ", clf.__str__()[:10], "...\n")
#
#     return accuracy_score(y_test, clf.predict(x_test))


def accuracy_precision_recall_metrics(clf, x_test, y_test):
    print("Calculation precision metrics for ", clf.__str__()[:10], "...\n")

    predictions = clf.predict(x_test)
    return accuracy_score(y_test, predictions), precision_score(y_test, predictions, average='macro'), \
           recall_score(y_test, predictions)


def test_datum(clf, datum):
    datum = list(map(float, datum.split('\t')))

    return clf.predict([datum]), clf.predict_proba([datum])


def main():
    print("Welcome.")

    file = input("Enter data file name: ")
    headers, labels, features, count, fraud_labels, fraud_features = get_data(file)

    ratio = float(input("Enter split ratio: "))
    x_train, x_test, y_train, y_test = split_data(features, labels, count, ratio)

    knn_clf = KNeighborsClassifier()
    tree_clf = tree.DecisionTreeClassifier()
    gaus_clf = GaussianNB()
    rf_clf = RandomForestClassifier()
    log_clf = LogisticRegression()
    mlp_clf = MLPClassifier()
    # svc_clf = SVC()

    knn_clf = train_clf(knn_clf, x_train, y_train)
    tree_clf = train_clf(tree_clf, x_train, y_train)
    gaus_clf = train_clf(gaus_clf, x_train, y_train)
    rf_clf = train_clf(rf_clf, x_train, y_train)
    log_clf = train_clf(log_clf, x_train, y_train)
    mlp_clf = train_clf(mlp_clf, x_train, y_train)
    # svc_clf = train_clf(svc_clf, x_train, y_train)

    # # ** TEST FRAUD RECOGNITION RATE ** #
    # knn_catch_rate = metrics(knn_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # tree_catch_rate = metrics(tree_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # gaus_catch_rate = metrics(gaus_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # rf_catch_rate = metrics(rf_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # log_catch_rate = metrics(log_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # mlp_catch_rate = metrics(mlp_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100
    # # svc_catch_rate = metrics(svc_clf, fraud_features, fraud_labels) / len(fraud_labels) * 100

    print("Tree classifier metrics: ", accuracy_precision_recall_metrics(tree_clf, x_test, y_test))
    print("KNN Classifier metrics: ", accuracy_precision_recall_metrics(knn_clf, x_test, y_test))
    print("Gaussian classifier metrics: ", accuracy_precision_recall_metrics(gaus_clf, x_test, y_test))
    print("Random Forest Classifier metrics: ", accuracy_precision_recall_metrics(rf_clf, x_test, y_test))
    print("Logistic Regression classifier metrics: ", accuracy_precision_recall_metrics(log_clf, x_test, y_test))
    print("MLP Neural-Net Regression classifier metrics: ", accuracy_precision_recall_metrics(mlp_clf, x_test, y_test))
    # print("SVC classifier catch rate: ", svc_catch_rate)


    #
    # knn_score = metrics(knn_clf, x_test, y_test)
    # tree_score = metrics(tree_clf, x_test, y_test)
    #
    # print("Tree classifier accuracy score: ", tree_score, " KNN Classifier accuracy score: ", knn_score)

    # while True:
    #     x = input("Enter test datum (q to quit): ")
    #
    #     if x == 'q':
    #         break
    #
    #     result, probability = test_datum(tree_clf, x)
    #     print("Tree Classifier result: {} with probability {}.".format(result, probability))
    #     result, probability = test_datum(knn_clf, x)
    #     print("KNN Classifier result: {} with probability {}.".format(result, probability))
    print("Bye")


if __name__ == "__main__":
    main()
