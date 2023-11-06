import numpy as np
import numpy.linalg
import os
import torch

pseudo_action = np.array([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
action = np.ones([1, 4])


print(pseudo_action.shape)
print(action.shape)

pseudo_action = np.concatenate((pseudo_action, action), axis=0)
print(pseudo_action.shape)
print(pseudo_action)