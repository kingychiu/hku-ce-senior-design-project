from preprocess import PreProcess

data = {}

with open('./datasets/switch_ag12bbc.txt', 'r', encoding='utf8') as f1:
    lines = f1.readlines()

    for i in range(len(lines)):
        line = lines[i]
        label = line.split('|sep|')[0]
        feature = line.split('|sep|')[1].split(',')
        # print(len(features[0]))  # 1536
        # print(len(features))
        if label in data.keys():
            data[label].append(feature)
        else:
            data[label] = [feature]
    f1.close()

labels = []
features = []
for key in data.keys():
    data[key] = data[key][:100]
    for d in data[key]:
        labels.append(key)
        features.append(('\t'.join(d)).replace('\n', ''))
    print(key, len(data[key]))
    print(len(data[key][1]))

print(len(labels))
print(len(features))
with open('./datasets/projection_labels.txt', 'w', encoding='utf8') as lf:
    lf.write('\n'.join(labels))

with open('./datasets/projection_features.txt', 'w', encoding='utf8') as ff:
    ff.write('\n'.join(features))
