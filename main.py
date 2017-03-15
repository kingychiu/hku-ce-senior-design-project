from preprocess import PreProcess

p = PreProcess('./datasets/ag_dataset_20000_each.txt')
labels, tensor = p.run_one_hot()

with open('./datasets/ag_dataset_20000_each_one_hot.txt', 'w', encoding='utf8') as f1:
    document_strs = []
    for matrix in tensor:
        char_strs = []
        for vector in matrix:
            char_str = ','.join(vector)
            char_strs.append(char_str)
        document_str = '|c|'.join(char_strs)
        document_strs.append(document_str)
    lines = []
    for i in range(len(document_strs)):
        lines.append(labels[i] + '|l|' + document_strs[i])
    f1.write('\n'.join(lines))
