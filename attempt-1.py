# easy attempt: tree SKLearn tree classifier

import csv
import time

import pydotplus
from sklearn import tree
from sklearn.externals.six import StringIO

print("getting data....")
with open('creditcardnotime.csv', 'rt') as fi:
    reader = csv.reader(fi)
    data = list(reader)
    header = data[0][0:29]
    labels = list()
    features = list()

    print("formatting and converting data...")
    number = 150000
    # get features and labels
    iterdata = iter(data);
    next(iterdata)  # skip the headers
    for index, value in enumerate(iterdata):
        if index > number:
            break
        labels.append(bool(int(value[29])))  # convert 0 to false and 1 to true
        features.append([float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list

print("Training decision tree... ")
begin_time = time.time();
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print("Decision tree training done! Training duration: ", time.time() - begin_time, " seconds.")
print("There were ", labels.count(True), " fraud occurrences in ", number, " training samples.")

# visualize graph
dot_data = StringIO()
tree.export_graphviz(clf,
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
    print(clf.predict(x), " with probability: ", clf.predict_proba(x))

