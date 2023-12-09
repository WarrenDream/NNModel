#Copyright(c) CL.Gao All rights reserved

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from einops import rearrange
from einops import repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual,self).__init__()
        self.fn = fn 
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) +x