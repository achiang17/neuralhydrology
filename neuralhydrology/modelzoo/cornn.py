from typing import Dict
from collections import defaultdict

import torch
from torch import nn

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

class coRNNCell(nn.Module):
    def __init__(self, n_inp, n_hid, dt, gamma, epsilon, intermediate_factor):
        super(coRNNCell, self).__init__()
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon
        self.intermediate_factor = intermediate_factor
        self.i2h = nn.Linear(intermediate_factor * n_inp + n_hid + n_hid, n_hid)
        self.i2m = nn.Linear(n_inp, intermediate_factor * n_inp)

    def forward(self, x, hy, hz):
        if self.intermediate_factor != 1:
            x = torch.tanh(self.i2m(x))
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
    module_parts = ['embedding_net', 'cell', 'head']

    def __init__(self, cfg: Config):
        super(coRNN, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)

        self.cell = coRNNCell(n_inp=self.embedding_net.output_size,
                                    n_hid=cfg.hidden_size,
                                    dt=cfg.dt,
                                    gamma=cfg.gamma,
                                    epsilon=cfg.eps,
                                    intermediate_factor=cfg.intermediate_factor)
        
        self.n_hid = cfg.hidden_size
        self.n_out = self.output_size

        # self.dropout = nn.Dropout(p=cfg.output_dropout)

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
        x_d = self.embedding_net(data) #[seq_len, batch_size, features]

        seq_len, batch_size, _ = x_d.size()

        hy = x_d.data.new(batch_size, self.n_hid).zero_()
        hz = x_d.data.new(batch_size, self.n_hid).zero_()

        output = defaultdict(list)
        for t in range(x_d.size(0)):
            hy, hz = self.cell(x_d[t], hy, hz)
            output['hy'].append(hy)
            output['hz'].append(hz)

        # stack to [batch_size, seq_len, hidden size]
        pred = {key: torch.stack(val,1) for key, val in output.items()}
        pred.update(self.head(pred['hy']))
        return pred
