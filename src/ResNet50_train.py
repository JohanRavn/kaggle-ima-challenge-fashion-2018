import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf
import numpy as np
from dataset import load_annotations, generate_sets, BatchGenerator#, generate_results_report
from utils import f_score
from sklearn.metrics import fbeta_score
from keras.applications import Xception, MobileNet, ResNet50, VGG16
from keras.utils import multi_gpu_model
from keras.layers import Dense, GlobalAveragePooling2D, Reshape, Dropout, Conv2D, Activation, Flatten
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import fbeta_score

def train():
    classes= 230
    with tf.device('/cpu:0'):
        base_model = ResNet50(
                        include_top=False,
                        weights=None,
                        input_shape=(224, 224, 3),
                        classes=classes,
                        )
        x = base_model.output
        x = Flatten()(x)
        x = Dense(classes, activation='sigmoid', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=x)
        # model.load_weights("../models/ResNet50_all_epoch_6.h5")
    parallel_model = multi_gpu_model(model, gpus=2)
    adam = Adam(lr=0.0001)
    parallel_model.compile(loss='categorical_crossentropy',
                           optimizer=adam,
                           metrics=['accuracy', f_score])

    train, valid = load_annotations("train", 0.00)
    # valid, _ = load_annotations("validation", 0.00)
    x_train_set, y_train_set = generate_sets('train', train)
    # x_valid_set, y_valid_set = generate_sets('train', valid)
    train_generator = BatchGenerator(x_train_set, y_train_set, 32, (224,224))
    # valid_generator = BatchGenerator(x_valid_set, y_valid_set, 32, (224,224))
    print(len(train_generator))
    for i in range(1, 20):
        parallel_model.fit_generator(train_generator,
                                     steps_per_epoch=31705,
                                     epochs=1,
                                     verbose=1,
                                     # validation_data=valid_generator,
                                     # validation_steps=1586,
                                     max_queue_size=200,
                                     workers=10,
                                     use_multiprocessing=True)

        model.save_weights("../models/ResNet50_all_DA_epoch_"+str(6+i)+".h5", overwrite=True)
if __name__ == "__main__":
    train()
