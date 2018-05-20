# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model
from dataset import load_test_data, load_validation_data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
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
from sklearn.utils import shuffle

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

def load_xception_model(weights):
    classes = 230
    base_model = Xception(
                    include_top=False,
                    weights=None,
                    input_shape=(299, 299, 3),
                    classes=classes,
                    )
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(classes, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(weights)
    return model

def load_resnet50_model(weights):
    classes= 230
    base_model = ResNet50(include_top=False, weights=None,
                          input_shape=(224, 224, 3), classes=classes)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(classes, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(weights)
    return model

def load_vgg16_model(weights):
    classes= 230
    base_model = VGG16(include_top=False, weights=None,
                          input_shape=(96, 96, 3), classes=classes)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(230, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=x)
    model.load_weights(weights)
    return model


def ensemble_xception(X, Y=None):
    pred = np.zeros((X.shape[0], 230))
    count = 0
    for i in range(0, 3):
        print("../models/xception_all_epoch_"+str(i)+".h5")
        model = load_xception_model("../models/xception_all_epoch_"+str(i)+".h5")
        pred += model.predict(X, verbose=1)
        count += 1
    pred /= count
    return pred


def ensemble_resnet50(X, Y=None):
    pred = np.zeros((X.shape[0], 230))
    count = 0
    for i in range(6, 13):
        print("loading model: ", "../models/ResNet50_all_epoch_"+str(i)+".h5")
        model = load_resnet50_model("../models/ResNet50_all_epoch_"+str(i)+".h5")
        tmp = model.predict(X, verbose=1)
        # check_different_thresholds(Y, tmp)
        pred += tmp
        count += 1
    pred /= count
    return pred

"""def ensemble_vgg16(X, Y):
    pred = np.zeros((X.shape[0], 230))
    count = 0
    for i in range(6, 15):
        print("loading model: ", "../models/vgg16_all_epoch_"+str(i)+".h5")
        model = load_resnet50_model("../models/vgg16_all_epoch_"+str(i)+".h5")
        pred += model.predict(X, verbose=1)
        count += 1
    pred /= count
    check_different_thresholds(Y, pred)
    return pred"""

def ensemble_test(X):
    pred = np.zeros((X.shape[0], 230))
    count = 0
    for i in range(6, 15):
        print("loading model: ", "../models/ResNet50_all_epoch_"+str(i)+".h5")
        model = load_model("../models/ResNet50_all_epoch_"+str(i)+".h5")
        pred += model.predict(X, verbose=1)
        count += 1
    pred /= count
    return pred

def check_different_thresholds(Y, pred):
    current = 0.0
    best = 0.0
    while current < 1.0:
        print("current threshold: ", current)
        tmp = np.copy(pred)
        tmp[tmp > current] = 1
        tmp[tmp <= current] = 0
        #print(classification_report(Y, tmp))
        print(accuracy_score(Y, tmp))
        print(fbeta_score(Y, tmp, 1, average='micro'))
        if fbeta_score(Y, tmp, 1, average='micro') > best:
            best = current
        current += 0.01


def validate():
    # X, Y = load_validation_data((299, 299), None)
    # pred = ensemble_xception(X, Y)
    # print("FB for xceptions")
    # check_different_thresholds(Y, pred)

    # pred[pred > 0.27] = 1
    # pred[pred <= 0.27] = 0
    # X, Y = load_validation_data((224, 224), None)
    # pred2 = ensemble_resnet50(X, Y)
    # check_different_thresholds(Y, pred2)
    # print("FB for xceptions + resnet50")

    # check_different_thresholds(Y, (pred + pred2)/2)
    # pred = (pred + pred2) / 2.0

    # pred2[pred2 > 0.25] = 1
    # pred2[pred2 <= 0.25] = 0

    # pred += pred2

    X, Y = load_validation_data((96, 96), 1000)
    model = load_vgg16_model("../models/vgg16_all_epoch_7.h5")
    pred3 = model.predict(X, verbose = 1)
    check_different_thresholds(Y, pred3)

    # pred3 = pred + pred2 + pred3
    # pred3 /= 2
    # check_different_thresholds(Y, pred3)
    pred3[pred3 > 0.26] = 1
    pred3[pred3 <= 0.26] = 0

    # pred = pred + pred2 + pred3
    # pred /= 3
    # pred[pred > 0.5] = 1
    # pred[pred <= 0.5] = 0
    # check_different_thresholds(Y, pred)

    print(classification_report(Y, pred3))
    print(accuracy_score(Y, pred3))
    print(fbeta_score(Y, pred3, 1, average='micro'))

       
def write_submission(pred, labels):
    #p = np.where(pred > 0.5)
    submission = open("submission_1.csv", "w+")
    submission.write("image_id,label_id\n")
    #for p, label in zip(pred, labels):
    for i in range(1, 39707):
        try:
            index = int(labels.index(str(i)))
            indices = list(np.where(pred[index] > 0.5)[0])
            #print(list(indices[0]))

            line = str(i) + ","
            for j in indices:
                #print(j)
                line += str(j)+" "
            line += "\n"
            submission.write(line)
        except ValueError:
            line = str(i) +",\n"
            submission.write(line)


    submission.close()


def test():
    X, labels = load_test_data((299, 299))
    pred = ensemble_xception(X)

    X, labels = load_test_data((224, 224))
    pred += ensemble_resnet50(X)
    pred /= 2.0
    # pred = ensemble_test(X)
    pred[pred > 0.26] = 1
    pred[pred <= 0.26] = 0
    write_submission(pred, labels)

if __name__ == '__main__':
    validate()