import sys
from lib.utils import read_training_file
from catboost import Pool, CatBoostClassifier

TRAIN_FILE_PATH = sys.argv[1]
print('Training File Path: {}'.format(TRAIN_FILE_PATH))

data, label = read_training_file(TRAIN_FILE_PATH)

train_pool = Pool(data, label)

model = CatBoostClassifier()
model.fit(train_pool)

model.save_model('model')
