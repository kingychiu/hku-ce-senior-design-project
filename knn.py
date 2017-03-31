from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

stat = {}

with open('./datasets/switch_ag12bbc.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    print(len(lines))
    labels = []
    features = []
    for line in lines:
        label = line.split('|sep|')[0]
        if label in stat.keys() and stat[label] < 2000:
            stat[label] += 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        elif label not in stat.keys():
            print(label)
            stat[label] = 1
            labels.append(label)
            features.append(line.split('|sep|')[1].split(','))
        else:
            pass
    f.close()

print(stat)

# x_train = features[:5000]
# x_test = features[5000:10000]
# y_train = labels[:5000]
# y_test = labels[5000:10000]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
classes = sorted(list(set(y_train)))
print(classes)
print('train', len(x_train))
print('test', len(x_test))
del labels
del features

neigh = KNeighborsClassifier(n_neighbors=50)
neigh.fit(x_train, y_train)

predictions = []
count = len(x_test)

summary = {}
for c in classes:
    summary[c] = {}
    for c1 in classes:
        summary[c][c1] = 0

for i in range(len(x_test)):
    sample = x_test[i]
    distances, neighbors = neigh.kneighbors(sample)
    neighbors = [y_train[n] for n in neighbors[0]]
    label_of_sample = y_test[i]
    for n in neighbors:
        summary[label_of_sample][n] += 1
    break

import operator
for c in summary.keys():
    ss = summary[c]
    print(c)
    for s in sorted(ss.items(), key=operator.itemgetter(1), reversed=True):
        if s[1] != 0:
            print('\t', s[0], s[1])



# for sample in x_test:
#     predictions.append(neigh.predict(sample))
#     count = count - 1
#     print(count)
# print(predictions[:10])
#
# t = 0
# t_by_class = {}
# total_by_class = {}
# for i in range(len(predictions)):
#     p = predictions[i][0]
#     if p in total_by_class.keys():
#         total_by_class[p] += 1
#     else:
#         total_by_class[p] = 1
#     if p == y_test[i]:
#         t += 1
#         if p in t_by_class.keys():
#             t_by_class[p] += 1
#         else:
#             t_by_class[p] = 1
#
# print(t / len(predictions))
# print()
# print('knn train data,', len(x_train))
# print('testing data', len(x_test))
# for k in list(t_by_class.keys()):
#     print(k, t_by_class[k] / total_by_class[k])
