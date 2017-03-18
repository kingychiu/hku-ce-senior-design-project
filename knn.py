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
x_train, x_test, y_train, y_test = train_test_split(features[:20000], labels[:20000], test_size=0.3,
                                                    random_state=42)
print(len(x_train))
print(len(x_train[0]))
print(len(x_test))
del labels
del features

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

predictions = []
count = len(x_test)
for sample in x_test:
    predictions.append(neigh.predict(sample))
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
    if p == y_test[i]:
        t += 1
        if p in t_by_class.keys():
            t_by_class[p] += 1
        else:
            t_by_class[p] = 1

print(t / len(predictions))
print(total_by_class)
for k in list(t_by_class.keys()):
    print(k, t_by_class[k] / total_by_class[k])
