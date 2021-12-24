"""
"""
import json
import sys

import torch
import esm
from deepTMpred.model import FineTuneEsmCNN, OrientationNet
from deepTMpred.utils import tmh_predict
from torch.utils.data import DataLoader
from deepTMpred.data import FineTuneDataset, batch_collate


def data_iter(data_path, pssm_dir, hmm_dir, batch_converter, label=False):
    data = FineTuneDataset(data_path, pssm_dir=pssm_dir, hmm_file=hmm_dir, label=label)
    test = DataLoader(data, len(data), collate_fn=batch_collate(batch_converter, label=label))
    return test


def test(model, orientation_model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for tokens, _ids, matrix, token_lengths in test_loader:
            tokens = tokens.to(device)
            results = model.esm(tokens, repr_layers=[12], return_contacts=False)
            token_embeddings = results["representations"][12][:, 1:, :]
            token_lengths = token_lengths.to(device)
            matrix = matrix.to(device)
            embedings = torch.cat((matrix, token_embeddings), dim=2)
            predict_list, prob = model.predict(embedings, token_lengths)
            orientation_out = orientation_model(embedings)
            predict = torch.argmax(orientation_out, dim=1)
            tmh_dict = tmh_predict(_ids, predict_list, prob, predict.tolist())
    return tmh_dict


def main():
    ###############
    test_file = sys.argv[3]
    tmh_model_path = sys.argv[1]
    orientation_model_path = sys.argv[2]
    device = torch.device('cpu')
    ###############

    model = FineTuneEsmCNN(768)
    pretrain_model, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model.add_module('esm', pretrain_model.to(device))
    model.load_state_dict(torch.load(tmh_model_path))
    model = model.to(device)

    orientation_model = OrientationNet()
    orientation_model.load_state_dict(torch.load(orientation_model_path))
    orientation_model = orientation_model.to(device)

    test_iter = data_iter(test_file, None, None, batch_converter, label=False)
    tmh_dict = test(model, orientation_model, test_iter, device)
    json.dump(tmh_dict, open('test.json', 'w'))


if __name__ == "__main__":
    main()
