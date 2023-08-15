import torchfcn
import torch
import torch.nn as nn
import numpy as np


class FCN8s(nn.Module):
    def __init__(self, num_class=1) -> None:
        self.model = torchfcn.models.FCN8s(n_class=21)
        self.model.load_state_dict(torch.load(torchfcn.models.FCN8s.download()))
        # our task
        self.model.score_fr = nn.Conv2d(4096, num_class, 1)
        self.model.score_pool3 = nn.Conv2d(256, num_class, 1)
        self.model.score_pool4 = nn.Conv2d(512, num_class, 1)
        self.model.upscore2 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False
        )
        self.model.upscore8 = nn.ConvTranspose2d(
            num_class, num_class, 16, stride=8, bias=False
        )
        self.model.upscore_pool4 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False
        )

    def forward(self, x: np.array) -> float:
        return self.model(x)
