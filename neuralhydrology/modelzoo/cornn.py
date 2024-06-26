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
    module_parts = ['embedding_net', 'cell', 'head']

    def __init__(self, cfg: Config):
        super(coRNN, self).__init__(cfg=cfg)

        self.embedding_net = InputLayer(cfg)
        self.input_size = self.embedding_net.statics_input_size + self.embedding_net.dynamics_input_size

        self.dt = 0.01
        self.gamma = 66
        self.epsilon = 15

        self.cell = coRNNCell(n_inp=self.input_size,
                                    n_hid=cfg.hidden_size,
                                    dt=self.dt,
                                    gamma=self.gamma,
                                    epsilon=self.epsilon)
        
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
        x_d = self.embedding_net(data)

        hy = torch.zeros(x_d.size(1), self.n_hid, device=x_d.device)
        hz = torch.zeros(x_d.size(1), self.n_hid, device=x_d.device)


        output_sequence = []
        for t in range(x_d.size(0)):
            hy, hz = self.cell(x_d[t], hy, hz)
            output_sequence.append(hy)
        output_sequence = torch.stack(output_sequence)
        output_sequence = output_sequence.transpose(0, 1)
        pred = self.head(output_sequence)

        return pred