"""
"""
import argparse
import torch
import esm
from deepTMpred.model import OrientationNet
import logging
from torch.utils.data import DataLoader, random_split
from deepTMpred.data import FineTuneDataset, orientation_batch_collate
from deepTMpred.utils import evaluate

#####################################
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# create a file handler
handler = logging.FileHandler('orientation.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#####################################


def get_params():
    parser = argparse.ArgumentParser('alpha-transmembrane protein orientation prediction')

    parser.add_argument('-b', '--batch-size', type=int,
                        default=16, help='mini-batch size (default: 16)')
    parser.add_argument('-n', '--num-epochs', type=int,
                        default=50, help='number of epochs (default: 50)')

    parser.add_argument('--l2', type=float, default=0.0001,
                        help='l2 regularize (default: 0.0001)')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')

    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout (default: 0.0)')

    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clipping max norm (default: 1.0)')

    parser.add_argument('--matrix', type=bool, default=False,
                        help='matrix (default: False)')

    args, _ = parser.parse_known_args()
    return args


def data_iter(data_path, pssm_dir, hmm_dir, batch_size, batch_converter, ratio=0.8, label=True):
    data = FineTuneDataset(data_path, pssm_dir=pssm_dir, hmm_file=hmm_dir)
    train_size = int(ratio * len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True,
                            collate_fn=orientation_batch_collate(batch_converter))
    val_iter = DataLoader(val_dataset, batch_size,
                          collate_fn=orientation_batch_collate(batch_converter))
    return train_iter, val_iter


def train(esm_model, model, optimizer, train_loader, epoch, device, pretrain_device, criterion):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    loss_accum = 0
    model.train()
    esm_model.eval()
    for tokens, labels, matrix, _ in train_loader:
        tokens = tokens.to(pretrain_device)
        optimizer.zero_grad()
        with torch.no_grad():
            results = esm_model(tokens, repr_layers=[34])
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        token_embeddings = results["representations"][34][:, 1:, :]
        token_embeddings = token_embeddings.to(device)
        matrix = matrix.to(device)
        embedding = torch.cat((matrix, token_embeddings), dim=2)
        labels = labels.to(device)
        output = model(embedding)
        loss = criterion(output, labels)
        loss.backward()
        # clip the gradient
        # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        loss_accum += loss.item()
        predict = torch.argmax(output, dim=1)

        predict_mask_p = predict == 1
        predict_mask_n = predict == 0
        tp += torch.sum((predict == labels).masked_select(predict_mask_p)).item()
        fp += torch.sum((predict != labels).masked_select(predict_mask_p)).item()
        tn += torch.sum((predict == labels).masked_select(predict_mask_n)).item()
        fn += torch.sum((predict != labels).masked_select(predict_mask_n)).item()

        total += labels.shape[0]
    print(tp, fp, tn, fn, flush=True)
    metric = evaluate(tp, fp, tn, fn)
    logger.info(
        "Train--Epoch:{:0>3d}, Loss:{:.5f}, F1:{:.5f}, ACC:{:.2%}, MCC:{:.5f}".format(epoch, loss_accum / total,
                                                                                      metric.f1_score,
                                                                                      metric.acc_score,
                                                                                      metric.mcc_score))


def test(esm_model, model, val_loader, epoch, device, pretrain_device, criterion):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    total = 0
    loss_accum = 0
    model.eval()
    esm_model.eval()
    with torch.no_grad():
        for tokens, labels, feature, _ in val_loader:
            tokens = tokens.to(pretrain_device)
            results = esm_model(tokens, repr_layers=[34])
            # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
            token_embeddings = results["representations"][34][:, 1:, :]
            token_embeddings = token_embeddings.to(device)
            feature = feature.to(device)
            embedding = torch.cat((feature, token_embeddings), dim=2)
            labels = labels.to(device)
            output = model(embedding)
            loss = criterion(output, labels)

            loss_accum = loss_accum + loss.item()
            predict = torch.argmax(output, dim=1)

            predict_mask_p = predict == 1
            predict_mask_n = predict == 0
            tp += torch.sum((predict ==
                             labels).masked_select(predict_mask_p)).item()
            fp += torch.sum((predict !=
                             labels).masked_select(predict_mask_p)).item()
            tn += torch.sum((predict ==
                             labels).masked_select(predict_mask_n)).item()
            fn += torch.sum((predict !=
                             labels).masked_select(predict_mask_n)).item()

            total += int(labels.shape[0])

        print(tp, fp, tn, fn, flush=True)
        metric = evaluate(tp, fp, tn, fn)
        logger.info(
            "Val--Epoch:{:0>3d}, Loss:{:.5f}, F1:{:.5f}, ACC:{:.2%}, MCC:{:.5f}".format(epoch, loss_accum / total,
                                                                                        metric.f1_score,
                                                                                        metric.acc_score,
                                                                                        metric.mcc_score))
    return loss_accum / total


def main(args):
    ###############
    train_file = "/home/wanglei/data/TMP/dataset/orientation.fa"
    pretrain = "/home/wanglei/.cache/torch/checkpoints/esm1_t34_670M_UR50S.pt"
    finetune_model_path = "/home/wanglei/data/TMP/project/src/save_model" \
                          "/esm_finetune_no_matrix_9_27_cnn_epoch_4_state_dict.pth "

    if args["matrix"]:
        pssm_dir = "/home/wanglei/data/TMP/dataset/orientation/pssm"
        hmm_dir = "/home/wanglei/data/TMP/dataset/orientation/hmm"
    else:
        pssm_dir, hmm_dir = None, None

    num_epochs = args["num_epochs"]
    dropout = args["dropout"]

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_device = torch.device('cuda:0')
    pretrain_device = torch.device('cuda:1')

    pretrain_model, alphabet = esm.pretrained.load_model_and_alphabet_local(pretrain)
    torch.cuda.empty_cache()
    pretrain_model = pretrain_model.to(pretrain_device)
    batch_converter = alphabet.get_batch_converter()

    model = OrientationNet(dropout=dropout)
    model_state_dict = model.state_dict()

    tmh_checkpoint = torch.load(finetune_model_path)
    finetune_model_dict = {k: v for k, v in tmh_checkpoint.items() if k in model_state_dict.keys()}
    model_state_dict.update(finetune_model_dict)
    model.load_state_dict(model_state_dict)
    model = model.to(model_device)

    # -------------
    logger.info("load dataset....")
    train_iter, val_iter = data_iter(train_file, pssm_dir, hmm_dir, args['batch_size'], batch_converter, ratio=0.8,
                                     label=True)
    # -------------

    ###############
    # optimizer
    lr = args['lr']
    l2 = args['l2']
    ignored_params = list(map(id, model.liner1.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    optimizer = torch.optim.Adam([{"params": base_params}, {"params": model.liner1.parameters(), "lr": 0.1 * lr}],
                                 lr=lr,
                                 weight_decay=l2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    nSamples = [227, 478]
    normedWeights = [1 - item / sum(nSamples) for item in nSamples]
    normedWeights = torch.FloatTensor(normedWeights).to(model_device)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum', weight=normedWeights)
    ###############

    for epoch in range(num_epochs):
        train(pretrain_model, model, optimizer, train_iter, epoch, model_device, pretrain_device, criterion)
        val_loss = test(pretrain_model, model, val_iter, epoch, model_device, pretrain_device, criterion)
        scheduler.step(val_loss)
        # test(pretrain_model, model, test_iter, epoch, model_device, pretrain_device, criterion)


if __name__ == "__main__":
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
