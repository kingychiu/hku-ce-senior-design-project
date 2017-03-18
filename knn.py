with open('./datasets/7blkup_4classes_dfeatures.txt', 'r', encoding='utf8') as f:
    lines = f.readlines()
    labels = []
    features = []
    for line in lines:
        labels.append(line.split('|sep|')[0])
        features.append(line.split('|sep|')[1].split(','))
    print(labels[:10])
    print(len(features[0]))
