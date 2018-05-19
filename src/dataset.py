import json
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.utils import Sequence
from keras.preprocessing.image import random_rotation, random_shear, random_shift

classes = [17, 19, 20, 53, 62, 66, 105, 106, 137, 138,
           153, 164, 171, 176, 214, 222]

num_classes = 230

def create_targets(anno):
    classes = []
    for cls in anno["labelId"]:
        classes.append(int(cls))

    return classes


def create_image(path):
    img = cv2.imread(path)

    try:
        img = cv2.resize(img, (224, 224))
    except cv2.error as e:
        #print(e)
        print(path)
        return None
    return img

def load_test_data():
    img_paths = glob.glob("../data/test_images/*")
    img_paths = sorted(img_paths)
    labels = []
    X = []
    print(len(img_paths))
    for i, path in enumerate(img_paths):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(img_paths)))
        img = create_image(path)
        labels.append(os.path.basename(path).strip('.jpeg'))
        X.append(img)
    print(labels)
    return np.array(X).astype('float32'), labels
    #return np.array(X).astype("float32"), labels


def create_sample(X_q, Y_q, current_annos, dataset):
    X = []
    Y = []
    for i, anno in (enumerate(current_annos)):
        if i % 100 == 0:
            print("loaded: ", i)
        targets = create_targets(anno)
        img_path = "../data/" + dataset + "_images/" + anno["imageId"] + ".jpeg"
        img = create_image(img_path)
        if img is None:
            continue
        X.append(img)
        Y.append(targets)
    X_q.put(X)
    Y_q.put(Y)

def convert_to_categorical(Y):
    new_y = []
    for y in Y:
        #print(y)
        tmp = np.zeros(num_classes)
        tmp[y] = 1
        #print(tmp)
        #y = tmp
        new_y.append(tmp)
    #for new in new_y:
    #    print(new)
    return new_y

def convert_to_specific_categorical(Y):
    new_Y = []
    for y in Y:
        categorical = []

        for i, current in enumerate(classes):
            if current in y:
                categorical.append(1)
            else:
                categorical.append(0)
        new_Y.append(categorical)
    #print("see conversion")
        #print(y)
        #print(categorical)
    return new_Y


def load_annotations(dataset_name, test_size):
    f = open("../input/" + dataset_name + ".json", "r")
    labels = json.loads(f.read())
    annotations = [x for x in labels["annotations"]]
    annotations = shuffle(annotations, random_state=0)

    train, valid = train_test_split(annotations,
                                    test_size=test_size,
                                    random_state=0)
    return train, valid

def generate_sets(dataset_name, annotations):
    x_set = []
    y_set = []
    for i, anno in enumerate(annotations):
        file_path = '../data/' + dataset_name + '_images/' + str(anno['imageId'] + '.jpeg')
        #target = set(classes).intersection(create_targets(anno))
        target = create_targets(anno)
        x_set.append(file_path)
        y_set.append(target)
        if i % 10000 == 0:
            print(i)
    y_set = convert_to_categorical(y_set)
    return x_set, y_set

class BatchGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def horizontal_flip(self, x):
        if np.random.random() < 0.5:
            axis = 1
            x = np.asarray(x).swapaxes(axis, 0)
            x = x[::-1, ...]
            x = x.swapaxes(0, axis)
        return x

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = []
        Y = []
        for file_name, y in zip(batch_x, batch_y):
            img = create_image(file_name)

            if img is None:
                continue

            img = self.horizontal_flip(img)
            img = random_rotation(img, 0.20)
            img = random_shift(img, 0.10, 0.10)
            img = random_shear(img, 0.10)
            X.append(img)
            Y.append(y)
        X = np.array(X).astype('float32')/255.0
        return np.array(X), np.array(Y)

def load_validation_data():
    f = open("../input/validation.json", "r")
    labels = json.loads(f.read())
    annotations = [x for x in labels["annotations"]]
    X = []
    Y = []
    for i, anno in enumerate(annotations):
        file_path = '../data/validation_images/' + str(anno['imageId'] + '.jpeg')
        #target = set(classes).intersection(create_targets(anno))
        target = create_targets(anno)
        img = create_image(file_path)
        if img is None:
            continue
        X.append(img)
        Y.append(target)
        if i % 1000 == 0:
            print(i)
    Y = convert_to_categorical(Y)
    print(np.array(Y).shape)
    return np.array(X).astype("float32"), np.array(Y)


def create_skip_list(x_set):
    f = open('skip_list.txt', 'w+')
    for i, path in enumerate(x_set):
        img = cv2.imread(path)
        if i % 10000 ==0:
            print(i)
        try:
            img = cv2.resize(img, (224, 224))
        except cv2.error as e:
            f.write(path+'\n')
            print(path) 
    f.close()

if __name__ == "__main__":
    train, valid = load_annotations(0.33)
    x_set, y_set = generate_sets(train)
    create_skip_list(x_set)
    #print(np.array(y_set).shape)
    #print(len(x_set))
    #train, valid = load_annotations(0.33)
    #print(len(train), len(valid))
    #trainGen = load_data()
    #validGen = BatchGenerator(valid, 1000)
    #gen = BatchGenerator(1000)
    #num_classes = gen.get_num_classes()
    #for i in range(10):
    #    X, y = trainGen.next()
    #    print(len(X), len(y))
