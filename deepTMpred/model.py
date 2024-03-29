import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from .crf import CRF
from typing import Dict, Iterable, Callable


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class FineTuneEsmCNN(nn.Module):
    def __init__(self, esm_dim, dropout=0.5):
        super(FineTuneEsmCNN, self).__init__()
        self.liner1 = nn.Linear(esm_dim + 50, 32)
        # self.bn1 = nn.BatchNorm1d(64)
        self.cov = nn.Conv1d(32, 16, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm1d(16)
        self.liner2 = nn.Linear(16, 2)
        self.dropout = dropout

        self.crf = CRF(2, batch_first=True)

    def _get_emission(self, x):
        x = F.dropout(self.liner1(x), p=self.dropout,
                      training=self.training)  # [batch, length, 1280]==>[batch, length, 64]
        x = x.permute(0, 2, 1).contiguous()
        x = self.cov(x)  # [batch, length, 64]==>[batch, length, 16]
        x = F.relu(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.liner2(x)
        return x

    def predict(self, x, tokens_length):
        x = self._get_emission(x)
        mask = length_to_mask(tokens_length, dtype=torch.bool)
        # loss = self.crf(x, labels, mask)
        out = self.crf.decode(x, mask)
        prob = []
        x = F.softmax(x, dim=2)
        for idx, item in enumerate(tokens_length.tolist()):
            prob.append(x[idx, :item, 1].tolist())
        return out, prob

    def forward(self, x, tokens_length, labels):
        x = self._get_emission(x)
        mask = length_to_mask(tokens_length, dtype=torch.bool)
        loss = self.crf(x, labels, mask)
        out = self.crf.decode(x, mask)
        return -loss, out


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output

        return fn

    def forward(self, x, x1, x2):
        _ = self.model(x, x1, x2)
        return self._features


class OrientationNet(nn.Module):
    def __init__(self, input_size=32, dropout=0.5):
        super(OrientationNet, self).__init__()

        # self.input_szie = input_size
        self.liner1 = nn.Linear(768 + 50, input_size)
        # self.cov = nn.Conv1d(64, 16, kernel_size=3, padding=1)
        # for i in self.cov.parameters():
        #     i.requires_grad = False
        self.w = nn.Parameter(torch.Tensor(input_size, 1))

        self.dense = nn.Linear(input_size, 8)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(8, 2)
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.dense.named_parameters():
            if 'weight' in name:
                init.xavier_normal_(param)
            elif 'bias' in name:
                init.normal_(param)
        nn.init.uniform_(self.w)

    def forward(self, x):
        # print("x", x.device)
        # print("w", self.w.device)
        # x = x[:,]
        x = F.relu(self.liner1(x))

        x = torch.cat((x[:, :5, :], x[:, -5:, :]), dim=1)  # [batch, 10, input_size]
        # x = torch.index_select(x, dim=1, index=topo)

        alpha = F.softmax(torch.matmul(x, self.w), dim=1)  # [batch, max_length, 1]

        out = x * alpha  # [batch, length, 64]
        out = torch.sum(out, 1)  # [batch, 64]
        # out = torch.mean(x, dim=1)
        # out = x[:, 0, :]
        # out = torch.mean(out, dim=1)
        out = F.relu(out)
        out = F.relu(self.dropout(self.dense(out)))  # [batch, 16]
        out = self.fc(out)
        # out = self.dense(out)
        return out
