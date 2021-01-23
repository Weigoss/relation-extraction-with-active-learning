import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    def __init__(self, config):
        """
        type_rnn: RNN, GRU, LSTM 可选
        """
        super(RNN, self).__init__()

        # self.xxx = config.xxx
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size
        self.num_layers = config.num_layers
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional
        self.last_layer_hn = config.last_layer_hn
        self.type_rnn = config.type_rnn

        rnn = eval(f'nn.{self.type_rnn}')
        self.rnn = rnn(input_size=self.input_size,
                       hidden_size=self.hidden_size,
                       num_layers=self.num_layers,
                       dropout=self.dropout,
                       bidirectional=self.bidirectional,
                       bias=True,
                       batch_first=True)

        # self._init_weights()

    def _init_weights(self):
        """orthogonal init yields generally good results than uniform init"""
        gain = 1  # use default value
        for nth in range(self.num_layers * self.bidirectional):
            # w_ih, (4 * hidden_size x input_size)
            nn.init.orthogonal_(self.rnn.all_weights[nth][0], gain=gain)
            # w_hh, (4 * hidden_size x hidden_size)
            nn.init.orthogonal_(self.rnn.all_weights[nth][1], gain=gain)
            # b_ih, (4 * hidden_size)
            nn.init.zeros_(self.rnn.all_weights[nth][2])
            # b_hh, (4 * hidden_size)
            nn.init.zeros_(self.rnn.all_weights[nth][3])

    def forward(self, x, x_len):
        """
        :param x: torch.Tensor [batch_size, seq_max_length, input_size], [B, L, H_in] 一般是经过embedding后的值
        :param x_len: torch.Tensor [L] 已经排好序的句长值
        :return:
        output: torch.Tensor [B, L, H_out] 序列标注的使用结果
        hn:     torch.Tensor [B, N, H_out] / [B, H_out] 分类的结果，当 last_layer_hn 时只有最后一层结果
        """
        B, L, _ = x.size()
        H, N = self.hidden_size, self.num_layers

        x = pack_padded_sequence(x, x_len, batch_first=True, enforce_sorted=True)
        output, hn = self.rnn(x)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=L)

        if self.type_rnn == 'LSTM':
            hn = hn[0]
        if self.bidirectional:
            hn = hn.view(N, 2, B, H).transpose(1, 2).contiguous().view(N, B, 2 * H).transpose(0, 1)
        else:
            hn = hn.transpose(0, 1)
        if self.last_layer_hn:
            hn = hn[:, -1, :]

        return output, hn


if __name__ == '__main__':
    class Config(object):
        type_rnn = 'LSTM'
        input_size = 5
        hidden_size = 4
        num_layers = 3
        dropout = 0.0
        last_layer_hn = False
        bidirectional = True


    config = Config()
    model = RNN(config)
    print(model)

    torch.manual_seed(1)
    x = torch.tensor([[4, 3, 2, 1], [5, 6, 7, 0], [8, 10, 0, 0]])
    x = torch.nn.Embedding(11, 5, padding_idx=0)(x)  # B,L,H = 3,4,5
    x_len = torch.tensor([4, 3, 2])

    o, h = model(x, x_len)

    print(o.shape, h.shape, sep='\n\n')
    print(o[-1].data, h[-1].data, sep='\n\n')
