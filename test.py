from utils.timers import Timer, CudaTimer
import torch
import numpy as np

a = torch.Tensor([[[[0, 0],
                    [1, 1]],

                   [[2, 2],
                    [2, 2]]]])

print(a.shape)
print(a[0, :, :, :])
print(torch.sum(a[0, -5:, :, :], dim=0))
