import os
import sys
import json

from sklearn.model_selection import KFold

database_path = '/Users/davidchoqueluqueroman/Documents/DATASETS_Local/RealLifeViolenceDataset/frames'

labels = os.listdir(database_path)
no_dir = os.path.join(database_path,'NonViolence')
fi_dir = os.path.join(database_path,'Violence')

no_data = os.listdir(no_dir)
fi_data = os.listdir(fi_dir)

kf = KFold(n_splits=5, shuffle=True)

k_no = kf.split(no_data)
k_fi = kf.split(fi_data)

print('no_data: ', no_data[77:97])
print('fi_data: ', fi_data[77:97])

# for each fold
i = 0
for (no_train_index, no_test_index), (fi_train_index, fi_test_index) in zip(k_no, k_fi):
    
    i += 1
    
    no_train, no_test = [no_data[x] for x in no_train_index], [no_data[x] for x in no_test_index]
    fi_train, fi_test = [fi_data[x] for x in fi_train_index], [fi_data[x] for x in fi_test_index]

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

    
    dst_json_path = database_path + str(i) + '.json'
    
    dst_data= {}
    
    dst_data['labels'] = labels
    
    dst_data['database'] = {}
    dst_data['database'].update(train_database)
    dst_data['database'].update(val_database)
    
    with open(dst_json_path, 'w') as dst_file:
            json.dump(dst_data, dst_file)
    