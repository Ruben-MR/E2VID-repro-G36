from model.model import E2VIDRecurrent
from utils.inference_utils import EventPreprocessor
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as img

flow = np.load("data/ecoco_depthmaps_test/train/sequence_0000000001/flow/disp01_0000000001.npy")
im1 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000000.png")
im2 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000001.png")

H, W = im1.shape
im1_torch = torch.from_numpy(im1.reshape((1, 1, im1.shape[0], im1.shape[1])))
flow = flow.reshape((1, flow.shape[0], flow.shape[1], flow.shape[2]))

xx = torch.arange(0, W).view(1, -1).repeat(1, 1, H, 1)
yy = torch.arange(0, H).view(-1, 1).repeat(1, 1, 1, W)
grid = torch.cat((xx, yy), 1).float()

grid = grid + flow
grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
grid = grid.float()
print(grid[0, :, 0, -1])
grid = grid.permute(0, 2, 3, 1)

output = F.grid_sample(im1_torch, grid)

"""
mask = torch.autograd.Variable(torch.ones(im1_torch.size()))
mask = F.grid_sample(mask, grid)
mask[mask < 0.9999] = 0
mask[mask > 0] = 1
output *= mask
"""

im1_hat = output.numpy()
plt.figure()
plt.imshow(im1)
plt.figure()
plt.imshow(im1_hat[0, 0, :, :])
plt.figure()
plt.imshow(im2)
plt.show()

