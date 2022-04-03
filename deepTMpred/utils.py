"""
utils
"""

import re
import esm
from argparse import Namespace


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


def load_model_and_alphabet_core(args_dict, regression_data=None):
    alphabet = esm.Alphabet.from_architecture(args_dict["args"].arch)

    # upgrade state dict
    pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
    prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
    model_args = {pra(arg[0]): arg[1] for arg in vars(args_dict["args"]).items()}
    model_type = esm.ProteinBertModel

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )
    return model, alphabet