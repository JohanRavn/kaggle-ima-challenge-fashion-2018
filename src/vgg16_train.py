import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from dataset import load_annotations, generate_sets, BatchGenerator#, generate_results_report
from sklearn.metrics import fbeta_score
from keras.applications import Xception, MobileNet, ResNet50, VGG16
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import fbeta_score
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

def fbeta_score_modified(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    fbeta_score(y_true, y_pred, 2, average='micro')

def f_score(y_true, y_pred):
    beta = 1
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


    #with tf.device('/cpu:0'):
    classes= 230
    with tf.device('/cpu:0'):
        base_model = VGG16(
                        include_top=False,
                        weights=None,
                        input_shape=(96, 96, 3),
                        #pooling='avg',
                        classes=classes,
                        )
        x = base_model.output
        x = Flatten()(x)
        x = Dense(512, activation='relu', name='fc1')(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dense(230, activation='sigmoid')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.load_weights("../models/vgg16_all_epoch_3.h5")
    # Replicates the model on 8 GPUs.
    # This assumes that your machine has 8 available GPUs.
    
    parallel_model = multi_gpu_model(model, gpus=2)
    #                                 cpu_merge=False)
    adam = Adam(lr=0.0001)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy', f_score])
    # This `fit` call will be distributed on 8 GPUs.
    # Since the batch size is 256, each GPU will process 32 samples.

    train, valid = load_annotations("train", 0.05)
    #valid, _ = load_annotations("validation", 0.00)
    x_train_set, y_train_set = generate_sets('train', train)
    x_valid_set, y_valid_set = generate_sets('train', valid)
    train_generator = BatchGenerator(x_train_set, y_train_set, 128, (96, 96))
    valid_generator = BatchGenerator(x_valid_set, y_valid_set, 128, (96, 96))
    print(len(train_generator))
    print(len(valid_generator))
    for i in range(4, 20):
        parallel_model.fit_generator(train_generator,
                                     steps_per_epoch=7530,
                                     # steps_per_epoch=30120,
                                     epochs=1,
                                     verbose=1,
                                     validation_data=valid_generator,
                                     validation_steps=397,
                                     # validation_steps=1586,
                                     max_queue_size=200,
                                     workers=10,
                                     use_multiprocessing=True)

        model.save_weights("../models/vgg16_all_epoch_"+str(i)+".h5", overwrite=True)
    #X_test, labels = load_test_data()
    #result = model.predict(X_test)
    #generate_results_report()

train()
