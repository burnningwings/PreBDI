import torch
from torch import nn

class TimeDistributed(nn.Module):

    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        # 对每个时间步需要做的运算
        self.module = module

        self.batch_first = batch_first

    def forward(self,x):
        if len(x.size()) <= 2:
            return self.module(x)

        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps
        x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1))
        y = self.module(x_reshape)

        # we have to reshape y
        if self.batch_first:
            # (samples, timesteps, output_size)
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # (timesteps, samples, output_size)
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y
