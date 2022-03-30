from model.model import E2VIDRecurrent
from utils.inference_utils import EventPreprocessor, CropParameters
from utils.ecoco_dataset import ECOCO_Train_Dataset
from utils.ecoco_sequence_loader import *
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


def pad_events(events, crop):
    origin_shape_events = events.shape
    events = events.unsqueeze(dim=2)
    events_after_padding = []
    for t in range(events.shape[0]):
        for item in range(events.shape[1]):
            event = events[t, item]
            event = crop.pad(event)
            events_after_padding.append(event)
    events = torch.stack(events_after_padding, dim=0)
    events = events.view(origin_shape_events[0], origin_shape_events[1], events.shape[1], events.shape[2], events.shape[3], events.shape[4]).squeeze(dim=2)
    return events


if __name__ == "__main__":
    # Model definition
    config = {'recurrent_block_type': 'convlstm', 'num_bins': 5, 'skip_type': 'sum', 'num_encoders': 3,
              'base_num_channels': 32, 'num_residual_blocks': 2, 'norm': 'BN', 'use_upsample_conv': True}
    model = E2VIDRecurrent(config=config).cuda()

    data_path = DATA_DIR
    train_dataset = ECOCO_Train_Dataset(sequence_length=3, start_index=0, path=data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

    mycrop = CropParameters(240, 180, model.num_encoders)

    for events, images, flows in train_loader:
        events = pad_events(events, mycrop)
        print(events.shape)
        events = events[:, :, :, mycrop.iy0:mycrop.iy1, mycrop.ix0:mycrop.ix1]
        print(events.shape)
        break

    """
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
    """
