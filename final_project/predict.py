import sys
from lib.utils import read_public_testing_file
from lib.utils import read_private_testing_file
from catboost import Pool, CatBoostClassifier

PUBLIC_FILE_PATH = sys.argv[1]
PRIVATE_FILE_PATH = sys.argv[2]

def predict(model, test_id, test_data, file_name):
    result = model.predict_proba(test_data)

    result = [(result[i][1], test_id[i]) for i in range(result.shape[0])]
    result.sort(reverse=True)
    with open(file_name, 'w') as output_file:
        output_file.write('Rank_ID\n')
        for prob, index in result:
            output_file.write(str(index) + '\n')

public_test_id, public_test_data = read_public_testing_file(PUBLIC_FILE_PATH)
private_test_id, private_test_data = read_private_testing_file(PRIVATE_FILE_PATH)

public_test_pool = Pool(public_test_data)
private_test_pool = Pool(private_test_data)

model = CatBoostClassifier()
model.load_model('model')

predict(model, public_test_id, public_test_data, './public.csv')
predict(model, private_test_id, private_test_data, './private.csv')
