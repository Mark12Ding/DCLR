import sys
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append('../backbone')
from r3d import generate_model

class GreatModel(nn.Module):
    def __init__(self, pretrain=None):
        super(GreatModel, self).__init__()
        self.backbone = generate_model(18)
        if pretrain is not None:
            self.backbone.load_state_dict(torch.load(pretrain, map_location="cpu"), strict=True)
        self.action_bn = nn.BatchNorm1d(512)
        self.action = nn.Linear(512, 101)
        self.action_bn.weight.data.fill_(1)
        self.action_bn.bias.data.zero_()
        
    def forward(self, seq):
        feat = self.backbone(seq)
        # feat = feat.mean(dim=(2,3,4))
        feat = F.normalize(feat)
        # return feat
        feat = self.action_bn(feat)
        # feat = self.dp(feat)
        logit = self.action(feat)
        return logit
