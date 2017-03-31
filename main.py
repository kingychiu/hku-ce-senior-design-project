from preprocess import PreProcess

p = PreProcess('./datasets/text-emotion-dataset.txt')
labels, tensor = p.run_look_up()
print(tensor[0])
with open('./datasets/text_emotion_7blkup.txt', 'w', encoding='utf8') as f1:
    document_strs = []
    for matrix in tensor:
        char_strs = []
        for vector in matrix:
            char_strs.append(vector)
        document_str = ','.join(char_strs)
        document_strs.append(document_str)
    lines = []
    for i in range(len(document_strs)):
        # if labels[i] != 'Entertainment':
        lines.append(labels[i] + '|l|' + document_strs[i])

    f1.write('\n'.join(lines))
