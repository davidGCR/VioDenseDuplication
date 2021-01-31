import os
import sys
import json

from sklearn.model_selection import train_test_split


def load_labels(database_path):
    if os.path.isdir(database_path):
        labels = os.listdir(database_path)

    return labels

def load_labels_rwf(database_path):
    # if os.path.isdir(database_path):
    #     labels = os.listdir(database_path)
    labels = ['fi','no']
    return labels

def get_rwf_dataset(database_path):
    if not os.path.exists(database_path):
        raise IOError('not exist path')

    train_database = {}

    train_fi_data = os.listdir(os.path.join(database_path, 'train', 'Fight'))
    train_no_data = os.listdir(os.path.join(database_path, 'train', 'NonFight'))

    for file_name in train_no_data:
        name = os.path.join('train', 'NonFight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'no'}
    for file_name in train_fi_data:
        name = os.path.join('train', 'Fight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'fi'}

    val_database = {}
    val_fi_data = os.listdir(os.path.join(database_path, 'val', 'Fight'))
    val_no_data = os.listdir(os.path.join(database_path, 'val', 'NonFight'))

    for file_name in val_no_data:
        name = os.path.join('val', 'NonFight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'validation'
        train_database[name]['annotations'] = {'label': 'no'}
    
    for file_name in val_fi_data:
        name = os.path.join('val', 'Fight', file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'validation'
        train_database[name]['annotations'] = {'label': 'fi'}
    
    return train_database, val_database


def get_dataset(database_path):
    if not os.path.exists(database_path):
        raise IOError('not exist path')
    no_dir = os.path.join(database_path, 'no')
    fi_dir = os.path.join(database_path, 'fi')

    no_data = os.listdir(no_dir)
    fi_data = os.listdir(fi_dir)

    no_train, no_test = train_test_split(no_data, test_size=0.2, shuffle=True)
    fi_train, fi_test = train_test_split(fi_data, test_size=0.2, shuffle=True)

    train_database = {}
    for file_name in no_train:
        name, _ = os.path.splitext(file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'no'}
    for file_name in fi_train:
        name, _ = os.path.splitext(file_name)
        train_database[name] = {}
        train_database[name]['subset'] = 'training'
        train_database[name]['annotations'] = {'label': 'fi'}

    val_database = {}
    for file_name in no_test:
        name, _ = os.path.splitext(file_name)
        val_database[name] = {}
        val_database[name]['subset'] = 'validation'
        val_database[name]['annotations'] = {'label': 'no'}
    for file_name in fi_test:
        name, _ = os.path.splitext(file_name)
        val_database[name] = {}
        val_database[name]['subset'] = 'validation'
        val_database[name]['annotations'] = {'label': 'fi'}

    return train_database, val_database


def generate_annotation(database_path, dst_json_path):
    labels = load_labels(database_path)
    train_database, val_database = get_dataset(database_path)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

def generate_annotation_rwf(database_path, dst_json_path):
    labels = load_labels_rwf(database_path)
    train_database, val_database = get_rwf_dataset(database_path)

    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)

if __name__ == '__main__':
    database_path = sys.argv[1]
    dst_json_path = database_path + '.json'

    # generate_annotation(database_path, dst_json_path)
    generate_annotation_rwf(database_path, dst_json_path)
