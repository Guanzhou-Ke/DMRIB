import torch
from torch import nn
import torch.nn.functional as F


class MIEstimator(nn.Module):
    
    def __init__(self, args, device) -> None:
        super().__init__()
        self.args = args
        self.device = device
        
        self.net = nn.Sequential(
            nn.Linear(self.args.hidden_dim + (self.args.vspecific.hidden_dims[-1]*4), 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1)
        )
        
    def forward(self, cz, vz):
        """
        cz: tensor, consistency representation.
        vz: tensor, view-specific representation.
        """
        pos = self.net(torch.cat([cz, vz], dim=1))
        neg = self.net(torch.cat([torch.roll(cz, 1, 0), vz], dim=1))
        return -F.softplus(-pos).mean() - F.softplus(neg).mean(), pos.mean() - neg.exp().mean() + 1