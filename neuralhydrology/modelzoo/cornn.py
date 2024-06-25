from typing import Dict

import torch
from torch import nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.i2h = nn.Linear(n_inp + n_hid + n_hid, n_hid)

    def forward(self, x, hy, hz):
        hz = hz + self.dt * (torch.tanh(self.i2h(torch.cat((x, hz, hy), 1)))
                             - self.gamma * hy - self.epsilon * hz)
        hy = hy + self.dt * hz
        return hy, hz

class coRNN(BaseModel):
    """coRNN model class.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    module_parts = ['embedding_net', 'coRNN_cell', 'head']

    def __init__(self, cfg: Config):
        super(coRNN, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.cell = coRNNCell(n_inp=self.output_size,
                                    n_hid=cfg.hidden_size,
                                    dt=cfg.dt,
                                    gamma=cfg.gamma,
                                    epsilon=cfg.epsilon)
        
        self.n_hid = cfg.hidden_size
        self.n_out = self.output_size

        self.dropout = nn.Dropout(p=cfg.output_dropout) # not sure if we need

        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the coRNN model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        x_d = self.embedding_net(data)

        hy = torch.zeros(x_d.size(1), self.n_hid, device=x_d.device)
        hz = torch.zeros(x_d.size(1), self.n_hid, device=x_d.device)

        output = torch.zeros(x_d.size(1), x_d.size(0), self.n_out, requires_grad=False)

        for t in range(x.size(0)):
            hy, hz = self.cell(x_d[t], hy, hz)
            output[:,t,:] = self.head(self.dropout(hy)) # include dropout?

        output = torch.squeeze(output)
        pred = {'y_hat': output}

        return pred
