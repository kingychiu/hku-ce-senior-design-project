from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

with open('./datasets/7blkup_4classes_dfeatures.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    labels = []
    features = []
    for line in lines:
        labels.append(line.split('|sep|')[0])
        features.append(line.split('|sep|')[1].split(','))
    print(labels[:10])
    print(len(features[0]))
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,
                                                    random_state=42)
print(len(x_train))
print(len(x_test))
del labels
del features
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)

predictions = []
for x in x_test:
    predictions.append(neigh.predict(x))
print(predictions[:10])
# for i in range(len(predictions)):
#