import torch
import torch.nn as nn
from utils import utils


class Autoregressor(nn.Module):
    def __init__(self, opt, input_size, hidden_dim):
        super(Autoregressor, self).__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=self.hidden_dim, batch_first=True
        )

        self.opt = opt

    def forward(self, input):  # input: B x L x C: eg. (22, 55, 512)

        cur_device = utils.get_device(self.opt, input)

        regress_hidden_state = torch.zeros(
            1, input.size(0), self.hidden_dim, device=cur_device) # (1, 22, 256)
        
        self.gru.flatten_parameters()
        output, regress_hidden_state = self.gru(input, regress_hidden_state)

        return output  # output: B x L x C: eg. (22, 55, 256)


if __name__ == 'main':
    opt = {'device': 'cuda'}
    autoregressor = Autoregressor(opt, input_size=512, hidden_dim=256)
    d = torch.randn(22, 55, 512)
    res = autoregressor(d)
    print(res.shape)
