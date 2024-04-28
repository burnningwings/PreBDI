import math

import torch
from einops import rearrange
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # position embeddings是不参与训练的
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # 注意：d_model需要是偶数
        #  0::2：从0开始，以步长为2进行取值，取到的都是偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)
        #  1::2：从1开始，以步长为2进行取值，取到的都是奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)

        # 改变一下shape,从原来的 (max_len, d_model)->  (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # register_buffer一下，作用是该组参数不会更新，但是保存模型时，该组参数又作为模型参数被保存
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x.size(1)是输入序列的长度（从max_len里截一段）
        # 返回的是(1, input_len, d_model)
        return self.pe[:, :x.size(1)]


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, freq='diw'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'hid':1, 'diw':2}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__()
        self.seg_len = seg_len

        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len=self.seg_len)
        x_embed = self.linear(x_segment)
        x_embed = rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b=batch, d=ts_dim)

        return x_embed

class DataEmbedding(nn.Module):
    def __init__(self, seg_len, d_model, freq='diw', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = DSW_embedding(seg_len=seg_len, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)

        return self.dropout(x)