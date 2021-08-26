import numpy as np
test_path = 'C:\\Users\\zhangpy\\Desktop\\seq2fold\\data\\astral_test.fa'
all_path = 'C:\\Users\\zhangpy\\Desktop\\seq2fold\\data\\astral_all.fa'


def getname(inf):
    names = []
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            name = line.split()[0][1:]
            names.append(name)
    return names


def fasta2dict(inf, label_mode=True):
    name = None
    data = {}
    for line in inf:
        line = line.strip()
        if line.startswith('>'):
            name = line.split()[0][1:]
            data[name] = {}
            data[name]['seq'] = ''
            if label_mode:
                label_split = line.split()[1].split('.')
                label = label_split[0]+'.'+label_split[1]
                data[name]['label'] = label
        else:
            data[name]['seq'] += line
    return data


if __name__ == '__main__':
    with open(all_path, 'r') as f:
        all_data = fasta2dict(f, label_mode=True)
    with open(test_path, 'r') as f:
        test_name = getname(f)
    labels = []
    count = 0  #
    for name in test_name:
        count += 1
        if count <= 237*2:
            labels.append('g.9')
        else:
            labels.append(all_data[name]['label'])
    import pandas as pd
    df = pd.DataFrame({'sample_id': test_name, 'category_id': labels})
    #df.to_csv('testset_with_label.csv',index=False)
    df.to_csv('test0825.csv', index=False)