import csv
import time

import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

print("getting data....")
with open('creditcardnotime.csv', 'rt') as csv_file:
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
        features.append([float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list
        count += 1

# splitting
split_ratio = .5
print("Splitting training & test data...")
feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=split_ratio)
print("There were ", target_train.count(True), " fraud occurrences in ", int(count * split_ratio), " training samples.")

# training tree
print("Training decision tree...")
begin_time = time.time()
tree_clf = tree.DecisionTreeClassifier()
tree_clf.fit(feature_train, target_train)
print("Decision tree training done! Training duration: ", time.time() - begin_time, " seconds.")

# tree metrics
print("Calculating accuracy metrics:")
tree_predictions = tree_clf.predict(feature_test)
tree_score = accuracy_score(target_test)
print("Tree accuracy: ", tree_score)

# training KNN
print("Training KNN...")
begin_time = time.time()
knn_clf = KNeighborsClassifier()
knn_clf.fit(feature_train, target_train)
print("Decision tree training done! Training duration: ", time.time() - begin_time, " seconds.")

# knn metrics
print("Calculating KNN accuracy metrics:")
knn_predictions = knn_clf.predict(feature_test)
knn_score = accuracy_score(target_test)
print("KNN accuracy: ", knn_score)

# visualize graph
dot_data = StringIO()
tree.export_graphviz(tree_clf,
                     out_file=dot_data,
                     feature_names=header,
                     filled=True, rounded=True,
                     impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("visualization.pdf")

# predict outcomes given inputs
while True:
    x = input("Enter test datum (q to quit): ")

    if x == "q":
        break

    x = list(map(float, x.split('\t')))
    print("Tree result: ", tree_clf.predict(x), " with probability: ", tree_clf.predict_proba(x))
    print("KNN result: ", knn_clf.predict(x), " with probability: ", knn_clf.predict_proba(x))
