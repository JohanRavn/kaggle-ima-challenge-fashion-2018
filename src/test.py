# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model
from dataset import load_test_data, load_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import numpy as np
from dataset import load_annotations, generate_sets, BatchGenerator#, generate_results_report
from sklearn.metrics import fbeta_score
from keras.applications import Xception, MobileNet, ResNet50, VGG16
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score, fbeta_score

def f_score(y_true, y_pred):
    beta = 2
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = (1 * beta**2) * precision * recall / ((beta**2) * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    print(f_score)
    return tf.reduce_mean(f_score)

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


def validate():
    X, Y = load_data("train", 10000)


    classes= 230
    base_model = ResNet50(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    #pooling='avg',
                    classes=classes,
                    )
    x = base_model.output
    x = Flatten()(x)
    #x = Dense(256, activation='relu', name='fc1')(x)
    #x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='sigmoid', name='predictions')(x)
    #pred = Dense(230, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights("../models/ResNet50_all_epoch.h5")


    pred = model.predict(X, batch_size=16, verbose=1)
    print(pred.shape)
    #for one, two in zip(Y, pred):
    #    print(one[66], two[66])
    #    two[66] = 1

    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0

    print(classification_report(Y, pred))
    print(accuracy_score(Y, pred))
    print(fbeta_score(Y, pred, 2, average='micro'))
    
    # create_submission(test_res, test_id, info_string)
def test():
    X, labels = load_test_data()

    print(len(X))

    classes= 230
    base_model = VGG16(
                    include_top=False,
                    weights=None,
                    input_shape=(224, 224, 3),
                    #pooling='avg',
                    classes=230,
                    )
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dense(256, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    #pred = Dense(230, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights("../models/100_epoch_vgg16.h5")


    preds = model.predict(X, batch_size=16, verbose=1)

    create_submission(test_res, test_id, info_string)

if __name__ == '__main__':
    validate()