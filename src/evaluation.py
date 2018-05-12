# -*- coding: utf-8 -*-
import panda as pd
from keras.models import load_model

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)




def evaluate():
    X, = load_test_data()

    print(len(X))

    model = load_model("../models/model.h5")

    preds = model.predict(X, batch_size=16, verbose=1)

    create_submission(test_res, test_id, info_string)


if __name__ == '__main__':
    evaluate()