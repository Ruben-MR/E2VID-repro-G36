import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from utils.loading_utils import load_model, get_device
import numpy as np
import argparse
import pandas as pd
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch, CropParameters, EventPreprocessor
from utils.inference_utils import IntensityRescaler
from utils.timers import Timer
from model.model import E2VIDRecurrent
import time
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options
import lpips

# Result plotter
def plot_training_data(train_losses, val_losses):
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title('training loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(train_losses)
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title('validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(val_losses)
    plt.grid()

    plt.show()


def flow_map(im, flo):
    """
    Flow mapping function, it wraps the previous image into the following timestep using the flowmap provided. The
    output will be the reconstructed image using the flow.
    :param im: tensor of shape (B, 1, H, W) containing the reconstructed images of the different batches at the previous
    time step
    :param flo: tensor of shape (B, 2, H, W) containing the flowmaps between the previous and the current/next timestep
    :return:
    """
    B, C, H, W = im.shape

    assert (im.is_cuda is True and flo.is_cuda is True) or (im.is_cuda is False and flo.is_cuda is False), \
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


# Custom loss function
def loss_fn(I_pred, I_pred_pre, I_true, first_iteration=False):
    # reconstruction loss
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    reconstruction_loss_fn = lpips.LPIPS(net='vgg').cuda()
    reconstruction_loss = reconstruction_loss_fn(I_pred, I_true)

    # temporal consistency loss
    if not first_iteration:
        alpha = 50  # hyper-parameter for weighting term (mitigate the effect of occlusions)
        # TODO: optical flow operator, in the lines below, it is not W*I_pred_pre but only W.
        #  In the paper it is a bit tricky to notice, but W is defined as a function and so,
        #  W(I_pred_pre) is not a matrix multiplication but a function W taking I_pred_pre as parameter
        W = 1
        weighting_term = torch.exp(-alpha * torch.linalg.norm(I_pred - W * I_pred_pre, ord=2, dim=(-1, -2)))
        temporal_loss = weighting_term * torch.linalg.norm(I_pred - W * I_pred_pre, ord=1, dim=(-1, -2))
    else:
        temporal_loss = 0

    # total loss
    lambda_ = 5  # weighting hyper-parameter
    loss = reconstruction_loss + lambda_ * temporal_loss

    return loss

# Training function
def training_loop(model, loss_fn, train_loader, validation_loader, lr=1e-4, epoch=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = [] # mean loss over each epoch
    val_losses = []  # mean loss over each epoch
    for e in range(epoch):
        # training process
        epoch_losses = []  # loss of each batch
        for x_batch, y_batch in train_loader:
            hidden_states = None
            I_predict_previous = None
            for t in range(x_batch.shape[0]):
                if t == 0:
                    I_predict, hidden_states = model(x_batch[t], None)
                    # print(x_batch[t].shape, I_predict.shape, y_batch[t].shape)
                    loss = loss_fn(I_predict, None, y_batch[t], first_iteration=True).sum()
                else:
                    I_predict, hidden_states = model(x_batch[t], hidden_states)
                    loss += loss_fn(I_predict, I_predict_previous, y_batch[t]).sum()
                # update variables
                I_predict_previous = I_predict
            # model update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())  # loss for single epoch
        train_losses.append(np.sum(epoch_losses))

        with torch.no_grad():
            epoch_losses = []  # loss of each batch
            for x_batch_val, y_batch_val in validation_loader:
                hidden_states = None
                I_predict_previous = None
                batch_loss = 0
                for t in range(x_batch_val.shape[0]):
                    if t == 0:
                        I_predict, hidden_states = model(x_batch_val[t], None)
                        loss = loss_fn(I_predict, None, y_batch_val[t], first_iteration=True).sum()
                    else:
                        I_predict, hidden_states = model(x_batch_val[t], hidden_states)
                        loss = loss_fn(I_predict, I_predict_previous, y_batch_val[t]).sum()
                    # update variables
                    I_predict_previous = I_predict
                    batch_loss += loss.item()
                epoch_losses.append(batch_loss)  # loss for single epoch
            val_losses.append(np.sum(epoch_losses))

    return train_losses, val_losses

# Event preprocessing options class
class PreProcessOptions:
    def __init__(self):
        self.no_normalize = False
        self.hot_pixels_file = None
        self.flip = False

use_pretrained = True

if __name__ == "__main__":
    # ======================================================================================================================================================
    # Model definition
    if not use_pretrained:
        config = {'recurrent_block_type': 'convlstm', 'num_bins': 5, 'skip_type': 'sum', 'num_encoders': 3,
                  'base_num_channels': 32,
                  'num_residual_blocks': 2, 'norm': 'BN', 'use_upsample_conv': True}
        model = E2VIDRecurrent(config=config)

        # Event preprocessor
        options = PreProcessOptions()
        preprocessor = EventPreprocessor(options)
    else:

        parser = argparse.ArgumentParser(
            description='Evaluating a trained network')
        parser.add_argument('-c', '--path_to_model', required=True, type=str,
                            help='path to model weights')
        parser.add_argument('-i', '--input_file', required=True, type=str)
        parser.add_argument('--fixed_duration', dest='fixed_duration', action='store_true')
        parser.set_defaults(fixed_duration=False)
        # parser.add_argument('-N', '--window_size', default=None, type=int,
        #                     help="Size of each event window, in number of events. Ignored if --fixed_duration=True")
        # parser.add_argument('-T', '--window_duration', default=33.33, type=float,
        #                     help="Duration of each event window, in milliseconds. Ignored if --fixed_duration=False")
        # parser.add_argument('--num_events_per_pixel', default=0.35, type=float,
        #                     help='in case N (window size) is not specified, it will be \
        #                           automatically computed as N = width * height * num_events_per_pixel')
        parser.add_argument('--skipevents', default=0, type=int)
        parser.add_argument('--suboffset', default=0, type=int)
        # parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
        # parser.set_defaults(compute_voxel_grid_on_cpu=False)

        set_inference_options(parser)

        args = parser.parse_args()

        # Read sensor size from the first line of the event file
        path_to_events = args.input_file

        header = pd.read_csv(path_to_events, delim_whitespace=True, header=None, names=['width', 'height'],
                             dtype={'width': np.int, 'height': np.int},
                             nrows=1)
        width, height = header.values[0]
        print('Sensor size: {} x {}'.format(width, height))

        # Load model
        model = load_model(args.path_to_model)
        device = get_device(args.use_gpu)

        model = model.to(device)
        # model.eval()
        #
        # reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

        N = 15119

        """ Read chunks of events using Pandas """
        initial_offset = args.skipevents
        sub_offset = args.suboffset
        start_index = initial_offset + sub_offset

        if args.fixed_duration:
            event_window_iterator = FixedDurationEventReader(path_to_events,
                                                             duration_ms=args.window_duration,
                                                             start_index=start_index)
        else:
            event_window_iterator = FixedSizeEventReader(path_to_events, num_events=N, start_index=start_index)

        with Timer('Processing entire dataset'):
            counter = 0
            for event_window in event_window_iterator:

                last_timestamp = event_window[-1, 0]

                with Timer('Building event tensor'):
                    event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height,
                                                                device=device)

                num_events_in_window = event_window.shape[0]
                start_index += num_events_in_window
                counter += 1
                if counter >= 1:
                    break

    # Do not worry about code above!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #============================================================================================================================================
    # ignore the code above, they are just used for taking out the event tensor and model
    # let's make a pseudo-dataset!!
    device = torch.device('cuda:0')
    events = event_tensor.unsqueeze(dim=0)
    # ==========================
    # pre-processing step here (normalizing and padding)
    crop = CropParameters(width, height, model.num_encoders)
    events = crop.pad(events) # (1, 5, 184, 240)
    # ==========================================

    events = events.view((1,*events.shape)) # (1, 1, 5, 184, 240)
    sequence_length = events.shape[0]
    batch_size = events.shape[1]
    events = events.tile((sequence_length, batch_size, 1, 1, 1)) # (sequence_len, batch_size, channel, H, W)
    events = events.to(device)
    labels = torch.rand((*events.shape)).detach()
    labels = labels[:, :, 0:1, :, :] # TODO: dealing with multiple channels
    labels = labels.to(device)

    train_loader = [(events, labels)]
    validation_loader = [(events.detach(), labels)]
    #============================================================================

    train_losses, val_losses = training_loop(model, loss_fn, train_loader, validation_loader)
    plot_training_data(train_losses, val_losses)

