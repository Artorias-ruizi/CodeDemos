import time
import datetime
import torch
from torch import nn


'''
Referenced from https://zhuanlan.zhihu.com/p/537552196
'''

if __name__ == '__main__':
    input_size = 5
    hid_size = 512
    pred_len = 1
    data = torch.rand(500, 90, 5)
    lstm = nn.LSTM(input_size=input_size, hidden_size=hid_size, num_layers=3, bias=True,
                   batch_first=True)
    linear = nn.Linear(hid_size, input_size, bias=True)
    # input data size: [batch_size, seq_len, input_size]
    # my input size should be: [8, 90, 5]
    # ================================================== #
    # input(batch_size, seq_len, input_size)
    # output(batch_size, seq_len, num_directions * hidden_size)

    output_lstm, (_, _) = lstm(data)
    output_linear = linear(output_lstm)
    output = output_linear[:, -1, :]
