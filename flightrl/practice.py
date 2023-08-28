import numpy as np
import torch

a2 = torch.tensor([0, 0, 0, 0])
print(a2.dtype)
a2 = a2.type(torch.float32)
print(a2.dtype)
