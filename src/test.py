# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model
from dataset import load_test_data, load_validation_data
from threshold import check_class_thresholds, threshold_test_set

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
from sklearn.utils import shuffle


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
                          input_shape=(192, 192, 3), classes=classes)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
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
    for i in range(0, 3):
        print("loading model: ", "../models/ResNet50_all_DA_epoch_"+str(i)+".h5")
        model = load_resnet50_model("../models/ResNet50_all_DA_epoch_"+str(i)+".h5")
        tmp = model.predict(X, verbose=1)
        pred += tmp
        count += 1
    pred /= count
    return pred

def ensemble_vgg16(X, Y=None):
    pred = np.zeros((X.shape[0], 230))
    count = 0
    for i in range(0, 6):
        print("loading model: ", "../models/vgg16_all_da_epoch_"+str(i)+".h5")
        model = load_vgg16_model("../models/vgg16_all_da_epoch_"+str(i)+".h5")
        tmp = model.predict(X, verbose=1)
        pred += tmp
        count += 1

    pred /= count
    return pred

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


def validate():
    X, Y = load_validation_data((299, 299))
    pred = ensemble_xception(X, Y)

    X, Y = load_validation_data((224, 224))
    pred2 = ensemble_resnet50(X, Y)
    pred += pred2

    X, Y = load_validation_data((192, 192))
    pred3 = ensemble_vgg16(X, Y)
    
    pred += pred3
    pred /=3.0
    check_class_thresholds(Y, pred)
    pred = threshold_test_set(pred)
    print(classification_report(Y, pred))
    print(accuracy_score(Y, pred))
    print(fbeta_score(Y, pred, 1, average='micro'))
       
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
    X, labels = load_test_data((299, 299), None)
    pred = ensemble_xception(X)
    X, labels = load_test_data((224, 224), None)
    pred += ensemble_resnet50(X)
    X, labels = load_test_data((192, 192), None)
    pred += ensemble_vgg16(X)
    pred /= 3.0

    pred = threshold_test_set(pred)

    write_submission(pred, labels)

if __name__ == '__main__':
    test()