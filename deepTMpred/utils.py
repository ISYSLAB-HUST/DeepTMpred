"""
utils
"""

import re
import math


def evaluate_h(item1, item2):
    tmh_H = 0
    for m in item1:
        for n in item2:
            if compare_h(m, n):
                tmh_H = tmh_H + 1
                break
    return tmh_H


def compare_h(p1, p2):
    _p_1_s, _p_1_e = p1
    _p_2_s, _p_2_e = p2
    cov = 2 * (min(_p_1_e, _p_2_e) - max(_p_1_s, _p_2_s)) >= max(_p_1_e - _p_1_s, _p_2_e - _p_2_s)
    if abs(_p_1_s - _p_2_s) <= 5 and abs(_p_1_e - _p_2_e) <= 5 and cov:
        return True
    else:
        return False


def precision_h(predict_list, observe_list):
    _p_r = 0
    _p_t = 0
    for label_1, label_2 in zip(predict_list, observe_list):
        tmp_i = []
        tmp_j = []
        predict = map(str, label_1)
        observe = map(str, label_2)
        for item in re.finditer(r'1+', ''.join(predict)):
            tmp_i.append(item.span())
        _p_t += len(tmp_i)
        for item in re.finditer(r'1+', ''.join(observe)):
            tmp_j.append(item.span())
        _p_r += evaluate_h(tmp_i, tmp_j)
    if _p_t == 0:
        return 0
    return _p_r / _p_t


def recall_h(predict_list, observe_list):
    _p_r = 0
    _p_t = 0
    for label_1, label_2 in zip(predict_list, observe_list):
        tmp_i = []
        tmp_j = []
        predict = map(str, label_1)
        observe = map(str, label_2)
        for item in re.finditer(r'1+', ''.join(predict)):
            tmp_i.append(item.span())
        for item in re.finditer(r'1+', ''.join(observe)):
            tmp_j.append(item.span())
        _p_t += len(tmp_j)
        _p_r += evaluate_h(tmp_i, tmp_j)
    if _p_t == 0:
        return 0
    return _p_r / _p_t


class evaluate(object):
    def __init__(self, tp, fp, tn, fn):
        self.tp = float(tp)
        self.fp = float(fp)
        self.tn = float(tn)
        self.fn = float(fn)

    @property
    def precision(self):
        if self.tp + self.fp == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        if self.tp + self.fn == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self):
        if self.precision + self.recall == 0:
            return 0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def acc_score(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def mcc_score(self):
        if (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn) == 0:
            return 0
        return (self.tp * self.tn - self.fp * self.fn) / math.sqrt(
            (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))


def tmh_predict(id_list, predict_str):
    tmh = {}
    for _id, predict in zip(id_list, predict_str):
        tmp = []
        predict = map(str, predict)
        for item in re.finditer(r'1+', ''.join(predict)):
            tmp.append(item.span())
        tmh[_id] = tmp
    return tmh


if __name__ == "__main__":
    list1 = [
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
    list2 = [
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
         0, 0, 0]]
    rec = recall_h(list1, list2)
    print("Recall: ", rec)
    pre = precision_h(list1, list2)
    print("Precision: ", pre)
    _id = ['a']
    tmh_dict = tmh_predict(_id, list1)
    print('tmh: ', tmh_dict)
