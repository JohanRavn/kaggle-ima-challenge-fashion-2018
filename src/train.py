import tensorflow as tf
import numpy as np
from dataset import load_data, load_test_data#, generate_results_report
from sklearn.metrics import fbeta_score
from keras.applications import Xception
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

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
    return tf.reduce_mean(f_score)

def train():
    X_train, y_train, num_classes = load_data("train", 50000)
    print(X_train.shape)
    print(y_train.shape)

    with tf.device('/cpu:0'):
        base_model = Xception(
                include_top=False,
                weights='imagenet',
                # input_shape=(299, 299, 3),
                # pooling='avg',
                classes=num_classes)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        pred = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=pred)

    # Replicates the model on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    parallel_model = multi_gpu_model(model,
                                      gpus=2,
                                      cpu_merge=True,
                                      cpu_relocation=True)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy', f_score])
    # This `fit` call will be distributed on 8 GPUs.
    # Since the batch size is 256, each GPU will process 32 samples.
    parallel_model.fit(X_train,
                       y_train,
                       epochs=20,
                       batch_size=32,
                       validation_split=0.3,
                       shuffle=True)


    model.save_model("../models/model.h5")
    #X_test, labels = load_test_data()
    #result = model.predict(X_test)
    #generate_results_report()

train()
