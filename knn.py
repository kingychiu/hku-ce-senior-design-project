from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

with open('./datasets/7blkup_5classes_dfeatures.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    labels = []
    features = []
    for line in lines:
        labels.append(line.split('|sep|')[0])
        features.append(line.split('|sep|')[1].split(','))
    print(labels[:10])
    print(len(features[0]))
    f.close()

x_train = features[:35000]
x_test = features[35000:35050]
y_train = labels[:35000]
y_test = labels[35000:35050]
classes = sorted(list(set(y_test)))
print('train', len(x_train))
print('test', len(x_test))
del labels
del features

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(x_train, y_train)

predictions = []
count = len(x_test)
for sample in x_test:
    proba = neigh.predict_proba(sample)[0]
    num_labels_each_data = 2
    p = [''] * num_labels_each_data
    max_i = 0
    print(len(proba))
    print(len(classes))
    for i in range(len(proba)):
        if proba[i] > proba[max_i]:
            p[1] = classes[max_i]
            max_i = i
            p[0] = classes[max_i]
    predictions.append(p)
    count = count - 1
    print(count)
print(predictions[:10])

t = 0
t_by_class = {}
total_by_class = {}
for i in range(len(predictions)):
    p = predictions[i][0]
    if p in total_by_class.keys():
        total_by_class[p] += 1
    else:
        total_by_class[p] = 1
    if y_test[i] in p:
        t += 1
        if p in t_by_class.keys():
            t_by_class[p] += 1
        else:
            t_by_class[p] = 1

print(t / len(predictions))
print()
print('knn train data,', len(x_train))
print('testing data', len(x_test))
for k in list(t_by_class.keys()):
    print(k, t_by_class[k] / total_by_class[k])
