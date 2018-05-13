import tensorflow as tf
import numpy as np
from dataset import load_annotations, generate_sets, BatchGenerator#, generate_results_report
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
    train, valid = load_annotations(0.33)
    x_train_set, y_train_set = generate_sets(train)
    x_valid_set, y_valid_set = generate_sets(valid)
    train_generator = BatchGenerator(x_train_set, y_train_set, 32)
    valid_generator = BatchGenerator(x_valid_set, y_valid_set, 32)
    print(len(train_generator))
    print(len(valid_generator))

    with tf.device('/cpu:0'):
        base_model = Xception(
                include_top=False,
                weights='imagenet',
                # input_shape=(299, 299, 3),
                # pooling='avg',
                classes=230)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        pred = Dense(230, activation='softmax')(x)
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

    parallel_model.fit_generator(train_generator,
                                 steps_per_epoch=1000,
                                 epochs=10,
                                 verbose=1,
                                 validation_data=valid_generator,
                                 validation_steps=300,
                                 max_queue_size=100,
                                 workers=4,
                                 use_multiprocessing=True)


    model.save_weights("../models/model.h5")
    #X_test, labels = load_test_data()
    #result = model.predict(X_test)
    #generate_results_report()

train()
