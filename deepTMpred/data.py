"""
dataset processing
"""
import re
import os
import random
from Bio import SeqIO
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


def setup_seed(seed):
    # Ref: https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


label_stoi = {'1': 0, 'H': 1}


def tokenize_label(label):
    # TMH and no-TMH label
    label = re.sub(r'[2]', '1', label)
    _token = []
    for res in label:
        _token.append(label_stoi[res])
    return torch.LongTensor(_token)


def parse_pssm(pssm_file):
    """
    scale (0, 1) -> (x-max)/(max-min)
    """
    with open(pssm_file, 'r') as fr:
        pssm_matrix = []
        for line in fr.readlines():
            split_line = line.split()
            if len(split_line) == 44:
                pssm_matrix.append([int(i) for i in split_line[2:22]])
        pssm_matrix = np.array(pssm_matrix).T
        pssm_matrix = MinMaxScaler().fit_transform(pssm_matrix).T
        return torch.as_tensor(pssm_matrix, dtype=torch.float32)


def parse_hmm(hmm_file):
    """
    parse hmm file to length*30 matrix
    """
    with open(hmm_file, 'r') as f:
        line = f.readline()
        while line[0] != '#':
            line = f.readline()
        # ignore the header
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        prob = []
        line = f.readline()
        while line[0:2] != '//':
            lineinfo = line.split()
            probs_ = [2 ** (-float(lineinfo[i]) / 1000) if lineinfo[i] != '*' else 0. for i in range(2, 22)]
            line = f.readline()
            lineinfo = line.split()
            extras_ = [2 ** (-float(lineinfo[i]) / 1000) if lineinfo[i] != '*' else 0. for i in range(0, 10)]
            prob.append(probs_ + extras_)
            line = f.readline()
            assert len(line.strip()) == 0  # ignore line
            line = f.readline()
        return torch.FloatTensor(prob)


class FineTuneDataset(Dataset):
    """custom dataset
    """

    def __init__(self, seq_path, pssm_dir=None, hmm_file=None, label=True):
        self.seq_path = seq_path
        self.pssm_dir = pssm_dir
        self.hmm_file = hmm_file
        self.label = label
        self.seq_record = SeqIO.index(self.seq_path, 'fasta')
        self.seq_keys = list(self.seq_record.keys())

    def __getitem__(self, index):
        _id = self.seq_keys[index]
        record = self.seq_record[_id]
        # fulfill 0 if pssm or hmm not exists
        record_length = int(len(record.seq) / 2) if self.label else len(record.seq)
        pssm_matrix = parse_pssm(os.path.join(
            self.pssm_dir, _id + '.pssm')) if self.pssm_dir else torch.zeros(record_length, 20)
        hmm_matrix = parse_hmm(os.path.join(
            self.hmm_file, _id + '.hmm')) if self.hmm_file else torch.zeros(record_length, 30)
        if self.label:
            seq_str = str(record.seq[:int(len(record.seq) / 2)])
            label_seq = tokenize_label(str(record.seq[int(len(record.seq) / 2):]))
            orientation = torch.tensor([1]) if record.description.split(' ')[1] == "True" else torch.tensor([0])
        else:
            seq_str = str(record.seq)
            return _id, seq_str, pssm_matrix, hmm_matrix, record_length
        return label_seq, seq_str, pssm_matrix, hmm_matrix, orientation, record_length

    def __len__(self):
        return len(self.seq_keys)


def batch_collate(batch_converter, label=True):
    def collate_fn(batch_data):
        """deepTMpred padding func

        Args:
            batch_data (list): tensor
        """
        batch_data.sort(key=lambda x: x[-1], reverse=True)
        seq_length = torch.LongTensor([x[-1] for x in batch_data])
        batch_list, matrix = [], []
        for item in batch_data:
            batch_list.append(item[:2])
            # pssm + hmm
            matrix.append(torch.cat(item[2:4], dim=1))
        matrix = pad_sequence(matrix, batch_first=True, padding_value=0)
        batch_item, _, batch_tokens = batch_converter(batch_list)
        # batch_item is "label" if label == True, batch_item is "sequence id"
        if label:
            batch_item = pad_sequence(batch_item, batch_first=True, padding_value=0)
            return batch_tokens, batch_item, matrix, seq_length
        else:
            return batch_tokens, batch_item, matrix, seq_length

    return collate_fn


def orientation_batch_collate(batch_converter, label=True):
    def collate_fn(batch_data):
        """deepTMpred padding func

        Args:
            batch_data (list): tensor
        """
        batch_data.sort(key=lambda x: x[-1], reverse=True)
        seq_length = torch.LongTensor([x[-1] for x in batch_data])
        batch_list, matrix, orientation = [], [], []
        for item in batch_data:
            batch_list.append(item[:2])
            # pssm + hmm
            matrix.append(torch.cat(item[2:4], dim=1))
            orientation.append(item[-2])
        matrix = pad_sequence(matrix, batch_first=True, padding_value=0)
        orientation = torch.cat(orientation)
        batch_item, _, batch_tokens = batch_converter(batch_list)
        # batch_item is "label" if label == True, batch_item is "sequence id"
        if label:
            return batch_tokens, orientation, matrix, seq_length
        else:
            return batch_tokens, batch_item, matrix, seq_length

    return collate_fn
