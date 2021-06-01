"""
utils
"""

import re


def tmh_predict(id_list, predict_str, prob, orientation):
    tmh = {}
    cutoff = 5
    for _id, predict, prob, ori in zip(id_list, predict_str, prob, orientation):
        tmp = []
        predict = map(str, predict)
        for item in re.finditer(r'1+', ''.join(predict)):
            if (item.end()-item.start()-1) >= cutoff:
                tmp.append([item.start()+1, item.end()])
        tmh[_id] = {'topo':tmp, 'topo_proba':prob, 'orientation':ori}
    return tmh

