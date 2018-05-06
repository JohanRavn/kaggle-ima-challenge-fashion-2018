import json
import cv2
import os
import glob
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue


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
        print(e)
        print(path)
        return None
    return img

def load_test_data():
    img_paths = glob.glob("../input/test_images/*")
    labels = []
    X = []
    for path in img_paths:
        img = create_image(path)
        labels.append(os.path.basename(path).strip('.jpeg'))
        X.append(img)
    return X, labels
    #return np.array(X).astype("float32"), labels


def create_sample(X_q, Y_q, current_annos):
    X = []
    Y = []
    for i, anno in (enumerate(current_annos)):
        if i % 100 == 0:
            print("loaded: ", i)
        targets = create_targets(anno)
        img_path = "../data/" + "train" + "_images/" + anno["imageId"] + ".jpeg"
        img = create_image(img_path)
        if img is None:
            continue
        X.append(img)
        Y.append(targets)
    X_q.put(X)
    Y_q.put(Y)

def convert_to_categorical(Y, num_classes):
    new_y = []
    for y in Y:
        tmp = np.zeros(num_classes + 1)
        tmp[y] = 1
        y = tmp
        new_y.append(tmp)
    return new_y

def calculate_num_classes(Y)
    unique_targets = set()
    for target in Y:
        unique_targets = unique_targets.union(target)
  
    num_classes = max(unique_targets)
    return num_classes

def load_data(dataset, selected):
    file_count = 15000
    f = open("../input/" + dataset + ".json", "r")
    labels = json.loads(f.read())
    anno = [x for x in labels["annotations"]][:file_count]

    extraced

    cpu_count = 14
    files_per_cpu = int(len(anno) / cpu_count) + 1
    process_list = []
    X_q = Queue()
    Y_q = Queue()
    Y = []
    for i in range(cpu_count):
        if i+1 == cpu_count:
            current_annos = anno[i*files_per_cpu:-1]
        else:
            current_annos = anno[i*files_per_cpu:(i+1)*files_per_cpu]
        p = Process(target=create_sample,
                    args=(X_q, Y_q, current_annos))
        p.start()
        process_list.append(p)

    X = []
    for i in range(cpu_count):
        X += X_q.get()
        Y += Y_q.get()

    num_classes = calculate_num_classes(num_classes)
    Y = convert_to_categorical(Y, num_classes)
    X = np.array(X).astype("float32")
    mean_pixel = [103.939, 116.779, 123.68]
    X[:, 0, :, :] -= mean_pixel[0]
    X[:, 1, :, :] -= mean_pixel[1]
    X[:, 2, :, :] -= mean_pixel[2]

    return X, np.array(Y), num_classes+1

if __name__ == "__main__":
    load_data("train")
