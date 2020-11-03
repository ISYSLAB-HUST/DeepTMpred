"""
main
"""
import os
import argparse
import torch
import esm
from deepTMpred.model import FineTuneEsmCNN
from deepTMpred.utils import recall_h, precision_h
import logging
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from deepTMpred.data import FineTuneDataset, batch_collate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('tmh_main.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_params():
    parser = argparse.ArgumentParser('alpha-transmembrane protein TMH model')

    parser.add_argument('-b', '--batch-size', type=int,
                        default=24, help='minibatch size (default: 24)')
    parser.add_argument('-n', '--num-epochs', type=int,
                        default=100, help='number of epochs (default: 100)')

    parser.add_argument('--l2', type=float, default=0.0,
                        help='l2 regularizer (default: 0.0)')

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clipping max norm (default: 1.0)')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout (default: 0.5)')

    parser.add_argument('--matirx', type=bool, default=False,
                        help='matirx (default: False)')
    args, _ = parser.parse_known_args()
    return args


def data_iter(data_path, pssm_dir, hmm_dir, batch_size, batch_converter, ratio=0.8, label=True):
    data = FineTuneDataset(data_path, pssm_dir=pssm_dir, hmm_file=hmm_dir)
    # construct random split
    train_size = int(ratio * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=batch_collate(batch_converter))
    val_iter = DataLoader(val_dataset, batch_size, collate_fn=batch_collate(batch_converter))
    return train_iter, val_iter


def test_data_iter(data_path, pssm_dir, hmm_dir, batch_converter, label=True):
    data = FineTuneDataset(data_path, pssm_dir=pssm_dir, hmm_file=hmm_dir)
    test_data = DataLoader(data, len(data), collate_fn=batch_collate(batch_converter))
    return test_data


def train(pretrain_model, model, optimizer, train_loader, epoch, device):
    correct = 0
    tmh_count = 0
    predict_val = 0
    aa_count = 0
    loss_accum = 0
    model.train()
    pretrain_model.eval()
    for tokens, labels, matrix, token_lengths in train_loader:
        # tokens = tokens.to(pretrained_device)
        with torch.no_grad():
            results = pretrain_model(tokens, repr_layers=[34])
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        token_embeddings = results["representations"][34][:, 1:, :]

        model.zero_grad()
        token_embeddings, labels, token_lengths = token_embeddings.to(device), labels.to(device), token_lengths.to(
            device)
        matrix = matrix.to(device)
        embeddings = torch.cat((matrix, token_embeddings), dim=2)

        loss, predict_list = model(embeddings, token_lengths, labels)
        loss.backward()
        optimizer.step()

        aa_count += torch.sum(token_lengths).item()
        loss_accum += loss.item()
        tmh_mask = (labels == 1)
        _predict = []
        for item in predict_list:
            _predict.append(torch.tensor(item))

        predict = pad_sequence(_predict, batch_first=True, padding_value=0)
        predict = predict.to(device)

        correct += torch.sum((predict == labels).masked_select(tmh_mask)).item()

        tmh_count += torch.sum(labels).item()  # tmh aa number

        predict_val += torch.sum(predict).item()

        # evaluate precision and recall
        observe_label = labels.tolist()
        p_h = precision_h(predict_list, observe_label)
        r_h = recall_h(predict_list, observe_label)
    try:
        f1 = 2 * correct * correct / (correct * tmh_count + correct * predict_val)
        logger.info(
            "Train--Epoch:{:0>3d}, Loss:{:.5f}, F1(r):{:.5f}, PRE(r):{:.2%}, REC(r):{:.2%}, PRE(H):{:.2%}, "
            "REC(H):{:.2%}".format(
                epoch, loss_accum / aa_count, f1, correct / predict_val, correct / tmh_count, p_h, r_h))
    except ZeroDivisionError as identifier:
        logger.error('error:{}'.format(identifier))
        logger.info("Train--Epoch:{:0>3d}, Loss:{:.5f}, correct(r):{}, predict_val(r):{}, tmh:{}".format(
            epoch, loss_accum / aa_count, correct, predict_val, tmh_count))


def val(pretrain_model, model, val_loader, device, epoch):
    correct = 0
    tmh_count = 0
    predict_val = 0
    aa_count = 0
    val_loss_accum = 0
    f1 = 0
    pretrain_model.eval()
    model.eval()
    with torch.no_grad():
        for tokens, labels, matrix, token_lengths in val_loader:
            # tokens = tokens.to(pretrained_device)
            results = pretrain_model(tokens, repr_layers=[34])
            token_embeddings = results["representations"][34][:, 1:, :]
            token_embeddings, labels, token_lengths = token_embeddings.to(device), labels.to(device), token_lengths.to(
                device)
            matrix = matrix.to(device)
            embeddings = torch.cat((matrix, token_embeddings), dim=2)

            loss, predict_list = model(embeddings, token_lengths, labels)

            # predict = model(token_seq, token_lengths)
            val_loss_accum += loss.item()
            aa_count += torch.sum(token_lengths).item()
            # label_mask = length_to_mask(label_lengths)
            tmh_mask = (labels == 1)
            _predict = []
            for item in predict_list:
                _predict.append(torch.tensor(item))

            predict = pad_sequence(_predict, batch_first=True)
            predict = predict.to(device)

            correct += torch.sum((predict == labels).masked_select(tmh_mask)).item()

            tmh_count += torch.sum(labels).item()  # tmh AA number
            predict_val += torch.sum(predict).item()

            # evaluate precision and recall
            observe_label = labels.tolist()
            p_h = precision_h(predict_list, observe_label)
            r_h = recall_h(predict_list, observe_label)

        try:
            f1 = 2 * correct * correct / (correct * tmh_count + correct * predict_val)
            logger.info(
                "Val--Epoch:{:0>3d}, Loss:{:.5f}, F1(r):{:.5f}, PRE(r):{:.2%}, REC(r):{:.2%}, PRE(H):{:.2%}, "
                "REC(H):{:.2%}".format(
                    epoch, val_loss_accum / aa_count, f1, correct / predict_val, correct / tmh_count, p_h, r_h))
        except ZeroDivisionError as identifier:
            logger.error('error:{}'.format(identifier))
            logger.info("Val--Epoch:{:0>3d}, Loss:{:.5f}, correct(r):{}, predict_val(r):{}, tmh:{}".format(
                epoch, val_loss_accum / aa_count, correct, predict_val, tmh_count))

    return f1, val_loss_accum / aa_count


def test(pretrain_model, model, test_loader, device, epoch):
    correct = 0
    tmh_count = 0
    predict_val = 0
    aa_count = 0
    test_loss_accum = 0
    pretrain_model.eval()
    model.eval()
    with torch.no_grad():
        for tokens, labels, matrix, token_lengths in test_loader:
            # tokens = tokens.to(pretrained_device)
            results = pretrain_model(tokens, repr_layers=[34])
            token_embeddings = results["representations"][34][:, 1:, :]

            token_embeddings, labels, token_lengths = token_embeddings.to(device), labels.to(device), token_lengths.to(
                device)

            matrix = matrix.to(device)
            embeddings = torch.cat((matrix, token_embeddings), dim=2)

            loss, predict_list = model(embeddings, token_lengths, labels)

            # predict = model(token_seq, token_lengths)
            test_loss_accum += loss.item()
            aa_count += torch.sum(token_lengths).item()
            # label_mask = length_to_mask(label_lengths)
            tmh_mask = (labels == 1)
            _predict = []
            for item in predict_list:
                _predict.append(torch.tensor(item))

            predict = pad_sequence(_predict, batch_first=True)
            predict = predict.to(device)

            correct += torch.sum((predict == labels).masked_select(tmh_mask)).item()

            tmh_count += torch.sum(labels).item()  # tmh AA number
            predict_val += torch.sum(predict).item()

            # evaluate precision and recall
            observe_label = labels.tolist()
            p_h = precision_h(predict_list, observe_label)
            r_h = recall_h(predict_list, observe_label)

        try:
            f1 = 2 * correct * correct / (correct * tmh_count + correct * predict_val)
            logger.info(
                "Test--Epoch:{:0>3d}, Loss:{:.5f}, F1(r):{:.5f}, PRE(r):{:.2%}, REC(r):{:.2%}, PRE(H):{:.2%}, "
                "REC(H):{:.2%}".format(
                    epoch, test_loss_accum / aa_count, f1, correct / predict_val, correct / tmh_count, p_h, r_h))
        except ZeroDivisionError as identifier:
            logger.error('error:{}'.format(identifier))
            logger.info("Test--Epoch:{:0>3d}, Loss:{:.5f}, correct(r):{}, predict_val(r):{}, tmh:{}".format(
                epoch, test_loss_accum / aa_count, correct, predict_val, tmh_count))


def main(args):
    ###############
    train_file = "./dataset/train_30.fa"
    test_file = "./dataset/test.fa"
    if args["matirx"]:
        pssm_dir = "/home/wanglei/data/TMP/dataset/feature/pssm/train"
        test_pssm_dir = "/home/wanglei/data/TMP/dataset/feature/pssm/test"
        hmm_dir = "/home/wanglei/data/TMP/dataset/feature/hmm/train"
        test_hmm_dir = "/home/wanglei/data/TMP/dataset/feature/hmm/test"
    else:
        pssm_dir, test_pssm_dir, hmm_dir, test_hmm_dir = None, None, None, None

    num_epochs = args["num_epochs"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###############

    logger.info("init model")
    pretrain_model, alphabet = esm.pretrained.pretrained.esm1_t34_670M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = FineTuneEsmCNN()
    model = model.to(device)

    # -------------
    logger.info("load dataset....")
    train_iter, val_iter = data_iter(train_file, pssm_dir, hmm_dir, args['batch_size'], batch_converter, ratio=0.85,
                                     label=True)
    test_iter = test_data_iter(test_file, test_pssm_dir, test_hmm_dir, batch_converter, label=True)
    # -------------

    ###############
    # optimizer
    lr = args['lr']
    l2 = args['l2']
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    ###############

    for epoch in range(num_epochs):
        # train epoch
        train(pretrain_model, model, optimizer, train_iter, epoch, device)
        val_f1, val_loss = val(pretrain_model, model, val_iter, device, epoch)
        scheduler.step(val_loss)
        test(pretrain_model, model, test_iter, device, epoch)


if __name__ == "__main__":
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
