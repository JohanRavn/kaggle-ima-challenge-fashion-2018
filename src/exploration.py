import json
import os
import glob
import numpy as np
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import matplotlib.image as image


def create_targets(anno):
    classes = []
    for cls in anno["labelId"]:
        classes.append(int(cls))

    return classes


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

def load_annotations(dataset):

    file_count = 30000
    f = open("../input/" + dataset + ".json", "r")
    labels = json.loads(f.read())
    anno = [x for x in labels["annotations"]][:file_count]
    return anno

def display_image_without(anno, label):
    for x in anno:
        #if str(label[0]) not in x['labelId'] and str(label[1]) in x['labelId']:
        if (str(label) in x['labelId']):
            try:
                img = plt.imread("../data/train_images/" + x['imageId'] + '.jpeg')
            except:
                print("failed to read:", "../data/train_images/" + x['imageId'] + '.jpeg')
                continue
            print(x['labelId'])
            plt.imshow(img)
            plt.show()

def calculate_distribution(anno):

    distribution = []
    for x in anno:
        #print(x)
        distribution += x['labelId']
    unique_labels = set(distribution)
    frequency = []
    for unique in unique_labels:
        frequency.append([unique, distribution.count(unique)])

    frequency = sorted(frequency, key=lambda x: x[1])
    for f in frequency:
        print(f)


if __name__ == "__main__":
    anno = load_annotations("train")
    calculate_distribution(anno)
    #display_image_without(anno, [100, 66])
    display_image_without(anno, 215)


# 51: bh/korset?
# 52: role/cosplay
# 53: ??
# 54: bh/korset?
# 55: ??
# 33: brown jeans?
# 100: tights
# 200: coat
# 86: lær?
# 68: shoes