import math
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, fbeta_score
from multiprocessing import Process, Queue

def check_different_thresholds(Y, pred):
    current = 0.0
    best = 0.0
    while current < 1.0:
        print("current threshold: ", current)
        tmp = np.copy(pred)
        tmp[tmp > current] = 1
        tmp[tmp <= current] = 0
        print(accuracy_score(Y, tmp))
        print(fbeta_score(Y, tmp, 1, average='micro'))
        if fbeta_score(Y, tmp, 1, average='micro') > best:
            best = current
        current += 0.01

def search_threshold(q, start, end, Y, pred):
    best = 0.0
    current = 0.0
    tmp = np.copy(pred)
    tmp[tmp > 0.5] = 1
    tmp[tmp <= 0.5] = 0
    for i in range(start, end):
        current = 0.0
        best_th = 0.0
        best = 0
        while current < 1.0:
            col = np.copy(pred[:, i])
            col[col > current] = 1
            col[col <= current] = 0
            tmp[:, i] = col

            if fbeta_score(Y, tmp, 1, average='micro') > best:
                best = fbeta_score(Y, tmp, 1, average='micro')
                best_th = current
            current += 0.01
        col = np.copy(pred[:, i])
        col[col > best_th] = 1
        col[col <= best_th] = 0
        tmp[:, i] = col
        q.put((i, best_th))


def search_multi_core(Y, pred):
    cpu_count = 14
    q = Queue()
    process_list = []
    cls_per_proc = math.ceil(230 / cpu_count)
    th_file = open("thresholds.txt", "w+")

    for i in range(cpu_count):
        # last cpu, corner case
        if i == cpu_count -1:
            p = Process(target=search_threshold,
                        args=(q, i*cls_per_proc, 230, Y, np.copy(pred)))
        else:
            p = Process(target=search_threshold,
                        args=(q, i*cls_per_proc, (i+1)*cls_per_proc, Y, np.copy(pred)))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    for i in range(230):
        th = q.get()
        th_file.write(str(th[0]) + "," + str(th[1]) + "\n")
    th_file.close()

def check_class_thresholds(Y, pred, thresholds=None):
    if thresholds is None:
        search_multi_core(Y, pred)
    th_file = open("thresholds.txt", "r+")
    for line in th_file:
        loc, th = line.strip("\n").split(",")
        loc = int(loc)
        th = float(th)
        col = pred[:, loc]
        col[col > th] = 1
        col[col <= th] = 0
        pred[:, loc] = col

    th_file.close()

def threshold_test_set(pred, thresholds=None):
    if thresholds is None:
        th_file = open("thresholds.txt", "r+")
    else:
        th_file = open(thresholds, "r+")
    for line in th_file:
        loc, th = line.strip("\n").split(",")
        loc = int(loc)
        th = float(th)
        col = pred[:, loc]
        col[col > th] = 1
        col[col <= th] = 0
        pred[:, loc] = col
    return pred
