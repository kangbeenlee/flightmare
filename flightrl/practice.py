import numpy as np
import numpy.linalg
import os
import torch

a = torch.arange(4, dtype=torch.float, requires_grad=True)
print(a)
a.log()[-1].backward()
print(a)
print(a.grad)