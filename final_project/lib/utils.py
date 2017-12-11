import numpy as np
import pandas

def one_hot(index, size):
    v = ""
    for i in range(size):
        if i == index:
            v += "1.0"
        else:
            v += "0.0"
        if i != size - 1:
            v += " "
    return v

def data_preprocess(file_name):
    print('data preprocessing...')
    print('file: {}'.format(file_name))
    data = pandas.read_csv(file_name, sep=',')
    title_list = list(data)

    with open(file_name + '_feature.txt', 'w') as output_file:
        for index, row in data.iterrows():
            for title in title_list:
                now_feature = row[title]
                if title == 'SEX':
                    output_file.write(str(now_feature - 1) + ' ')
                elif title =='EDUCATION':
                    output_file.write(one_hot(now_feature, 7) + ' ')
                elif title == 'MARRIAGE':
                    output_file.write(one_hot(now_feature, 4) + ' ')
                elif title[:3] == 'PAY':
                    output_file.write(one_hot(now_feature + 2.0, 11) + ' ')
                elif title == 'Y':
                    output_file.write(str(float(now_feature)))
                else:
                    output_file.write(str(now_feature) + ' ')
            output_file.write('\n')

def read_file(file_path):
    data_preprocess(file_path)
    input_file = open(file_path + '_feature.txt', 'r').readlines()
    input_file = [seq.strip().split(' ')[1:] for seq in input_file]
    input_file = [[float(value) for value in seq] for seq in input_file]
    return input_file

def read_training_file(file_path):
    training_file = read_file(file_path)
    data = [seq[:-1] for seq in training_file]
    label = [seq[-1] for seq in training_file]
    return np.array(data), np.array(label)

def read_testing_file(file_name):
    testing_file = read_file(file_name)
    return np.array(testing_file)

def read_public_testing_file(file_path):
    return [i for i in range(1, 5001)], read_testing_file(file_path)

def read_private_testing_file(file_path):
    return [i for i in range(5001, 10001)], read_testing_file(file_path)
