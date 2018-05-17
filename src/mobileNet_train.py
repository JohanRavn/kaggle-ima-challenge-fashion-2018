import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import numpy as np
from dataset import load_annotations, generate_sets, BatchGenerator#, generate_results_report
from sklearn.metrics import fbeta_score
from keras.applications import Xception, MobileNet
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation
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
    train, valid = load_annotations(0.20)
    x_train_set, y_train_set = generate_sets(train)
    x_valid_set, y_valid_set = generate_sets(valid)
    train_generator = BatchGenerator(x_train_set, y_train_set, 16)
    valid_generator = BatchGenerator(x_valid_set, y_valid_set, 16)
    print(len(train_generator))
    print(len(valid_generator))

    #with tf.device('/cpu:0'):
    classes= 230
    base_model = MobileNet(
                    include_top=False,
                    weights="imagenet",
                    # input_shape=(299, 299, 3),
                    #pooling='avg',
                    classes=230,
                    alpha=0.75
                    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    shape = (1, 1, int(1024 * 0.75))
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(1e-3, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = GlobalAveragePooling2D()(x)
    #pred = Dense(230, activation='softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    model = Model(inputs=base_model.input, outputs=x)
        #model.load_weights("../models/model.h5")
    # Replicates the model on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    # parallel_model = multi_gpu_model(model,
    #                                  gpus=2,
    #                                 cpu_merge=False)
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy', f_score])
    # This `fit` call will be distributed on 8 GPUs.
    # Since the batch size is 256, each GPU will process 32 samples.

    model.fit_generator(train_generator,
                                 steps_per_epoch=500,
                                 epochs=100,
                                 verbose=1,
                                 validation_data=valid_generator,
                                 validation_steps=1000,
                                 max_queue_size=100,
                                 workers=4,
                                 use_multiprocessing=True)


    model.save("../models/mobilenet.h5", overwrite=True)
    #X_test, labels = load_test_data()
    #result = model.predict(X_test)
    #generate_results_report()

train()
