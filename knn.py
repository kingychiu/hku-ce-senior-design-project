from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

with open('./datasets/7blkup_4classes_dfeatures.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    labels = []
    classes = sorted(list(set(labels)))
    features = []
    for line in lines:
        labels.append(line.split('|sep|')[0])
        features.append(line.split('|sep|')[1].split(','))
    print(labels[:10])
    print(len(features[0]))

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(features[:5000], labels[:5000])
y_test = labels[5000:10000]
predictions = []
count = 5000
t = 0
t_by_class = {}
total_by_class = {}
for i in range(5000, 10000):
    sample = features[i]
    p = neigh.predict(sample)[0]
    if p in total_by_class.keys():
        total_by_class[p] += 1
    else:
        total_by_class[p] = 1
    if p == y_test[i]:
        t += 1
        if p in t_by_class.keys():
            t_by_class[p] += 1
        else:
            t_by_class[p] = 1
    count = count - 1
    print(count)

print(t / len(y_test))
print()
print('knn train data,', len(features[:5000]))
print('testing data', len(features[5000:]))
for k in list(t_by_class.keys()):
    print(k, t_by_class[k] / total_by_class[k])
