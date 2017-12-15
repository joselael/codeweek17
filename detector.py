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
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE


class Detector:

    def visualize_graph(self, clf, header, output):
        dot_data = StringIO()
        tree.export_graphviz(clf,
                             out_file=dot_data,
                             feature_names=header,
                             filled=True, rounded=True,
                             impurity=False)

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf(output)

    def get_data(self, file):
        print("getting data...")
        with open(file, 'rt') as csv_file:
            reader = csv.reader(csv_file)
            data = list(reader)
            header = data[0][0:29]
            targets = list()
            features = list()
            fraud_features = list()
            fraud_targets = list()

            print("formatting and converting data...\n")

            # get features and targets
            count = 0
            iterdata = iter(data)
            next(iterdata)  # skip the headers
            for index, value in enumerate(iterdata):

                targets.append(bool(int(value[29])))  # convert 0 to false and 1 to true

                if targets[index]:
                    fraud_targets.append(True)
                    fraud_features.append([float(i) for i in value[0:29]])

                features.append(
                    [float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list
                count += 1

        return header, targets, features, count, fraud_targets, fraud_features

    def split_data(self, features, targets, count, ratio):
        # x = features, y = targets
        print("Splitting training & test data...")
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=ratio)
        print("There were ", y_train.count(True), " fraud occurrences in ", int(count * ratio), " training samples.\n")

        return x_train, x_test, y_train, y_test

    def over_sample(self, x, y):
        sm = SMOTE()

        return sm.fit_sample(x, y)

    def train_clf(self, clf, x_train, y_train):
        print("Training classifier ", clf.__str__()[:10], "...")
        begin_time = time.time()
        clf.fit(x_train, y_train)
        print("Training done! Training duration: ", time.time() - begin_time, " seconds.\n")

        return clf

    # def accuracy_metrics(clf, x_test, y_test):
    #     print("Calculating accuracy metrics for ", clf.__str__()[:10], "...\n")
    #
    #     return accuracy_score(y_test, clf.predict(x_test))

    def accuracy_precision_recall_metrics(self, clf, x_test, y_test):
        print("Calculating precision metrics for ", clf.__str__()[:10], "...")

        predictions = clf.predict(x_test)
        return accuracy_score(y_test, predictions), precision_score(y_test, predictions, average='macro'), \
               recall_score(y_test, predictions)

    def test_datum(self, clf, datum):
        datum = list(map(float, datum.split('\t')))

        return clf.predict([datum]), clf.predict_proba([datum])

    def test_transaction(self, clf, file):
        with open(file, 'rt') as file:
            content = file.read()

            return self.test_datum(clf, content)

    def __init__(self):
        print("Welcome.")

        file = input("Enter data file name: ")
        headers, targets, features, count, fraud_targets, fraud_features = self.get_data(file)

        ratio = float(input("Enter split ratio: "))
        x_train, x_test, y_train, y_test = self.split_data(features, targets, count, ratio)

        # ** Over-sample training data using smote ** #
        x_train, y_train = self.over_sample(x_train, y_train)

        # knn_clf = KNeighborsClassifier()
        tree_clf = tree.DecisionTreeClassifier()
        gaus_clf = GaussianNB()
        rf_clf = RandomForestClassifier()
        log_clf = LogisticRegression()
        # mlp_clf = MLPClassifier()
        # svc_clf = SVC()
        # eclf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rf_clf), ('gnb', gaus_clf),
        #                                     ('tree', tree_clf)], voting='hard', n_jobs=-1)

        # for clf, label in zip([log_clf, rf_clf, gaus_clf, tree_clf, eclf],
        #                       ['Logistic Regression', 'Random Forest', 'naive Bayes',
        #                        'Tree', 'Ensemble']):
        #     scores = cross_val_score(clf, features, targets, cv=3, scoring='accuracy')
        #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

        # knn_clf = train_clf(knn_clf, x_train, y_train)
        # tree_clf = self.train_clf(tree_clf, x_train, y_train)
        # gaus_clf = self.train_clf(gaus_clf, x_train, y_train)
        # rf_clf = self.train_clf(rf_clf, x_train, y_train)
        log_clf = self.train_clf(log_clf, x_train, y_train)
        # mlp_clf = train_clf(mlp_clf, x_train, y_train)
        # svc_clf = train_clf(svc_clf, x_train, y_train)
        # eclf = self.train_clf(eclf, x_train, y_train)

        # # ** TEST FRAUD RECOGNITION RATE ** #
        # knn_catch_rate = accuracy_precision_recall_metrics(knn_clf, fraud_features, fraud_targets)[0] / len(fraud_targets) * 100
        # tree_catch_rate = self.accuracy_precision_recall_metrics(tree_clf, fraud_features, fraud_targets)[0] / len(
        #     fraud_targets) * 100
        # gaus_catch_rate = self.accuracy_precision_recall_metrics(gaus_clf, fraud_features, fraud_targets)[0] / len(
        #     fraud_targets) * 100
        # rf_catch_rate = self.accuracy_precision_recall_metrics(rf_clf, fraud_features, fraud_targets)[0] / len(
        #     fraud_targets) * 100
        # log_catch_rate = self.accuracy_precision_recall_metrics(log_clf, fraud_features, fraud_targets)[0] / len(
        #     fraud_targets) * 100
        # mlp_catch_rate = accuracy_precision_recall_metrics(mlp_clf, fraud_features, fraud_targets)[0] / len(fraud_targets) * 100
        # # svc_catch_rate = metrics(svc_clf, fraud_features, fraud_targets) / len(fraud_targets) * 100

        # eclf_catch_rate = self.accuracy_precision_recall_metrics(eclf, fraud_features, fraud_targets)[0] / len(
        #     fraud_targets) * 100

        # # print("KNN classifier catch rate: ", knn_catch_rate)
        # print("Tree classifier catch rate: ", tree_catch_rate)
        # print("Gaussian classifier catch rate: ", gaus_catch_rate)
        # print("Random Forest classifier catch rate: ", rf_catch_rate)
        # print("Log Regression classifier catch rate: ", log_catch_rate)
        # # print("MLP NN classifier catch rate: ", mlp_catch_rate)
        # print("Ensemble NN classifier catch rate: ", eclf_catch_rate)

        # print("Tree classifier metrics: ", accuracy_precision_recall_metrics(tree_clf, x_test, y_test))
        # print("KNN Classifier metrics: ", accuracy_precision_recall_metrics(knn_clf, x_test, y_test))
        # print("Gaussian classifier metrics: ", accuracy_precision_recall_metrics(gaus_clf, x_test, y_test))
        # print("Random Forest Classifier metrics: ", accuracy_precision_recall_metrics(rf_clf, x_test, y_test))
        # print("Logistic Regression classifier metrics: ", accuracy_precision_recall_metrics(log_clf, x_test, y_test))
        # print("MLP Neural-Net Regression classifier metrics: ", accuracy_precision_recall_metrics(mlp_clf, x_test, y_test))
        # print("Ensemble Classifier metrics: ", accuracy_precision_recall_metrics(eclf, x_test, y_test))

        # print("SVC classifier catch rate: ", svc_catch_rate)

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

        print("Bye!")

    # if __name__ == "__main__":
    #     main()


myDetector = Detector()
