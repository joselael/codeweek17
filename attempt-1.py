# easy attempt: tree SKLearn tree classifier

import csv

from sklearn import tree

print("getting data....")
with open('data0.csv', 'rt') as fi:
    reader = csv.reader(fi)
    data = list(reader)
    header = data[0][0:29]
    labels = list()
    features = list()

    print("formatting and converting data...")
    # get labels
    iterdata = iter(data);
    next(iterdata)  # skip the headers
    for index, value in enumerate(iterdata):
        labels.append(bool(int(value[29])))  # convert 0 to false and 1 to true
        features.append([float(i) for i in value[0:29]])  # convert the first 28 elements of the line to a float list

print("Training decision tree... ")
clf = tree.DecisionTreeClassifier()
clf.fit(features, labels)
print("Decision tree training done!")

# predict outcomes given inputs
while True:
    x = input("Enter test datum (q to quit): ")

    if x == "q":
        break

    x = list(map(float, x.split('\t')))
    print(clf.predict([x]))

# print(features[0:1000])
# print(labels[0:1000])
# print(labels[0:50000].count(True)) # how many occurrences of fraud are there in the subset?

# print(feature[0])
# print(data[99999])
