import tensorflow as tf
import numpy as np
from dataset import load_data, load_test_data#, generate_results_report
from sklearn.metrics import fbeta_score

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

class Metrics(tf.keras.callbacks.Callback):
    """
    Custom class to log some more metrics during after each epoch of training.
    Currenlty supports snesitivity and specificity
    """
    def on_epoch_end(self, batch, logs={}):

        Y_val = self.model.validation_data[1][0]
        pred = self.model.predict(self.model.validation_data[0], batch_size=32)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        print("f1_score ", f1_score(Y_val, pred))


def train():
    X_train, y_train, num_classes = load_data("train")
    print(X_train.shape)
    print(y_train.shape)
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg',
        classes=num_classes)
    x = base_model.output
    pred = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=pred)

    adam = tf.keras.optimizers.Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss="categorical_crossentropy",
                  metrics=['accuracy', f_score])
    model.fit(x=X_train,
              y=y_train,
              batch_size=32,
              epochs=40,
              verbose=1,
              validation_split=0.3,
              shuffle=True
              )
    tf.keras.models.save_model(model, "../models/model")
    #X_test, labels = load_test_data()
    #result = model.predict(X_test)
    #generate_results_report()

train()
