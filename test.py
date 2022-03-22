from model.model import E2VIDRecurrent
from utils.inference_utils import EventPreprocessor
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as img


def flow_map(im, flo):
    B, C, H, W = im.shape
    assert (im.is_cuda is True and flo.is_cuda is True) or (im.is_cuda is False and flo.is_cuda is False),\
        "both tensors should be on the same device"
    assert C == 1, "the image tensor has more than one channel"
    assert flo.shape[1] == 2, "flow tensor has wrong dimensions"
    xx = torch.arange(0, W).view(1, -1).repeat(1, 1, H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, 1, 1, W)
    xx = xx.repeat(B, 1, 1, 1)
    yy = yy.repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    if im.is_cuda:
        grid = grid.cuda()
    vgrid = torch.autograd.Variable(grid) + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(im.double(), vgrid)
    """
    mask = torch.autograd.Variable(torch.ones(im.size())).cuda()
    mask = F.grid_sample(mask.double(), vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    output *= mask
    """
    return output


if __name__ == "__main__":
    flow = np.load("data/ecoco_depthmaps_test/train/sequence_0000000001/flow/disp01_0000000001.npy")
    im1 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000000.png")
    im2 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000001.png")

    flow_1 = np.load("data/ecoco_depthmaps_test/train/sequence_0000000000/flow/disp01_0000000001.npy")
    im1_1 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000000/frames/frame_0000000000.png")
    im2_1 = img.imread("data/ecoco_depthmaps_test/train/sequence_0000000000/frames/frame_0000000001.png")

    flow = flow.reshape((1, flow.shape[0], flow.shape[1], flow.shape[2]))
    flow_1 = flow_1.reshape((1, flow_1.shape[0], flow_1.shape[1], flow_1.shape[2]))
    flows = torch.concat((torch.from_numpy(flow), torch.from_numpy(flow_1)), dim=0).cuda()

    im1_torch = torch.from_numpy(im1.reshape((1, 1, im1.shape[0], im1.shape[1])))
    im1_1_torch = torch.from_numpy(im1_1.reshape((1, 1, im1.shape[0], im1.shape[1])))
    images = torch.concat((im1_torch, im1_1_torch), dim=0).cuda()

    images_reconstructed = flow_map(images, flows)

    im_hat = images_reconstructed.cpu().numpy()

    plt.figure()
    plt.imshow(im1)
    plt.figure()
    plt.imshow(im_hat[0, 0, :, :])
    plt.figure()
    plt.imshow(im2)
    plt.figure()
    plt.imshow(im1_1)
    plt.figure()
    plt.imshow(im_hat[1, 0, :, :])
    plt.figure()
    plt.imshow(im2_1)
    plt.show()
