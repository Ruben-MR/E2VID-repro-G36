import numpy as np
import matplotlib.image as img
import torch
from utils.train_utils import flow_map
import matplotlib.pyplot as plt


if __name__ == "__main__":
    flow = np.load("../data/ecoco_depthmaps_test/train/sequence_0000000001/flow/disp01_0000000001.npy")
    im1 = img.imread("../data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000000.png")
    im2 = img.imread("../data/ecoco_depthmaps_test/train/sequence_0000000001/frames/frame_0000000001.png")

    flow_1 = np.load("../data/ecoco_depthmaps_test/train/sequence_0000000000/flow/disp01_0000000001.npy")
    im1_1 = img.imread("../data/ecoco_depthmaps_test/train/sequence_0000000000/frames/frame_0000000000.png")
    im2_1 = img.imread("../data/ecoco_depthmaps_test/train/sequence_0000000000/frames/frame_0000000001.png")

    flow = flow.reshape((1, flow.shape[0], flow.shape[1], flow.shape[2]))
    flow_1 = flow_1.reshape((1, flow_1.shape[0], flow_1.shape[1], flow_1.shape[2]))
    flows = torch.concat((torch.from_numpy(flow), torch.from_numpy(flow_1)), dim=0).cuda()

    im1_torch = torch.from_numpy(im1.reshape((1, 1, im1.shape[0], im1.shape[1])))
    im1_1_torch = torch.from_numpy(im1_1.reshape((1, 1, im1.shape[0], im1.shape[1])))
    images = torch.concat((im1_torch, im1_1_torch), dim=0).double().cuda()

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