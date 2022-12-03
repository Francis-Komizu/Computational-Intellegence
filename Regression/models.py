import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, input_dim):
        super(Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def build_model(config):
    predictor = Predictor(input_dim=config.model.input_dim)

    optimizer = torch.optim.Adam([{'params': predictor.parameters(), 'initial_lr': config.train.learning_rate}],
                                 lr=config.train.learning_rate,
                                 weight_decay=config.train.weight_decay,
                                 betas=config.train.betas,
                                 eps=config.train.eps)

    return predictor, optimizer

