import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.inference_utils import CropParameters


class PreProcessOptions:
    """
    Event preprocessing options class
    """
    def __init__(self):
        self.no_normalize = False
        self.hot_pixels_file = None
        self.flip = False


class RescalerOptions:
    """
    Intensity rescaler options class
    """
    def __init__(self):
        self.auto_hdr = True
        self.auto_hdr_median_filter_size = 10
        self.Imin = None
        self.Imax = None


def plot_training_data(tr_losses, v_losses):
    """
    Function for plotting the training and validation loss evolution over the iterations
    :param tr_losses: list of losses over the iterations
    :param v_losses: list of losses on the validation set
    :return: nothing
    """
    plt.style.use('seaborn')
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.title('training loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(tr_losses)
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title('validation loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(v_losses)
    plt.grid()

    plt.show()


def pad_all(model, events, images):
    width = events.shape[-1]
    height = events.shape[-2]
    origin_shape_events = events.shape
    origin_shape_images = images.shape
    # # ==========================
    # pre-processing step here (normalizing and padding)
    crop = CropParameters(width, height, model.num_encoders)
    events = events.unsqueeze(dim=2)
    images = images.unsqueeze(dim=2)
    images = images.unsqueeze(dim=2)
    events_after_padding = []
    images_after_padding = []
    for t in range(events.shape[0]):
        for item in range(events.shape[1]):
            event = events[t, item]
            image = images[t, item]
            events_after_padding.append(crop.pad(event))
            images_after_padding.append(crop.pad(image))
    events = torch.stack(events_after_padding, dim=0)
    images = torch.stack(images_after_padding, dim=0)
    events = events.view(origin_shape_events[0], origin_shape_events[1], events.shape[1], events.shape[2], events.shape[3], events.shape[4]).squeeze(dim=2)
    images = images.view(origin_shape_images[0], origin_shape_images[1], images.shape[1], images.shape[2], images.shape[3], images.shape[4]).squeeze(dim=2)

    return events, images


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

    # Create a meshgrid with pixel locations
    xx = torch.arange(0, W).view(1, -1).repeat(1, 1, H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, 1, 1, W)
    xx = xx.repeat(B, 1, 1, 1)
    yy = yy.repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1)

    # Move the tensor to cuda if flow and images so are
    if im.is_cuda:
        grid = grid.cuda()
    # Change the positions of the pixels indexed in the grid tensor by using the flow
    vgrid = torch.autograd.Variable(grid) + flo

    # Normalize to range [-1, 1] for usage of grid_sample
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    # Permute to get the correct dimensions for the sampling function
    vgrid = vgrid.permute(0, 2, 3, 1)
    # Sample points from the previuos image according to the indexing grid
    output = F.grid_sample(im.double(), vgrid)
    """
    mask = torch.autograd.Variable(torch.ones(im.size())).cuda()
    mask = F.grid_sample(mask.double(), vgrid)
    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1
    output *= mask
    """
    return output


def loss_fn(I_pred, I_pred_pre, I_true, I_true_pre, reconstruction_loss_fn, flow=None, first_iteration=False):
    """
    Custom loss function as specified by the authors, takes the current and last predicted and ground-truth images
    and computes the loss function with perceptual and temporal consistency components using a value of 50 for the alpha
    weighing constant of the temporal consistency loss and lambda value of 5 for weighing both components of the loss
    :param I_pred: latest predicted image
    :param I_pred_pre: predicted image at the previous timestep
    :param I_true: ground-truth image of the latest prediction
    :param I_true_pre: ground-truth image of the previous timestep
    :param first_iteration: boolean for skipping the temporal consistency loss if the loss is being computed for the
    first timestep of the sequence
    :param flow: flow tensor between the previous and the current timestep of the sequence
    :return: value of the loss function
    """
    # reconstruction loss
    # image should be RGB, IMPORTANT: normalized to [-1,1]
    reconstruction_loss = reconstruction_loss_fn(I_pred, I_true)

    # temporal consistency loss
    if not first_iteration:
        alpha = 50  # hyper-parameter for weighting term (mitigate the effect of occlusions)
        # TODO: verify correct working
        if flow is not None:
            # Computation of the flow map operation upon the predicted images
            W = flow_map(I_pred_pre, flow)
            # Computation of the weighing term using the previous and current ground-truth images
            M = torch.exp(-alpha * torch.linalg.norm(I_true - flow_map(I_true_pre, flow), ord=2, dim=(-1, -2)))
        else:
            W, M = 1, 1
        # Compute the temporal loss
        temporal_loss = M * torch.linalg.norm(I_pred - W, ord=1, dim=(-1, -2))
    else:
        temporal_loss = 0

    # total loss
    lambda_ = 5  # weighting hyper-parameter
    loss = reconstruction_loss + lambda_ * temporal_loss

    return loss


# Training function
def training_loop(model, loss_fn, train_loader, validation_loader, rec_fun, lr=1e-4, epoch=5):
    """
    Function for implementing the training loop of the network
    :param model: network to be trained
    :param loss_fn: loss function to be used for backpropagation
    :param train_loader: data loader
    :param validation_loader: validation data loader
    :param lr:learning rate
    :param epoch:number of epochs of the training
    :return: list of training and validation losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []   # mean loss over each epoch
    val_losses = []  # mean loss over each epoch
    # Start iterating for the specific number of epochs
    for e in range(epoch):
        epoch_losses = []  # loss of each batch
        # Load the current data batch
        for x_batch, y_batch in train_loader:
            hidden_states = None
            I_predict_previous = None
            # Iterate over the timesteps (??)
            for t in range(x_batch.shape[0]):
                if t == 0:
                    I_predict, hidden_states = model(x_batch[t], None)
                    # print(x_batch[t].shape, I_predict.shape, y_batch[t].shape)
                    loss = loss_fn(I_predict, None, y_batch[t], None, rec_fun, first_iteration=True).sum()
                else:
                    I_predict, hidden_states = model(x_batch[t], hidden_states)
                    loss += loss_fn(I_predict, I_predict_previous, y_batch[t], y_batch[t - 1], rec_fun).sum()
                # update variables
                I_predict_previous = I_predict
            # model update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())  # loss for single epoch
        train_losses.append(np.sum(epoch_losses))

        # After every epoch, perform validation
        with torch.no_grad():
            epoch_losses = []  # loss of each batch
            # Load the data
            for x_batch_val, y_batch_val in validation_loader:
                hidden_states = None
                I_predict_previous = None
                batch_loss = 0
                for t in range(x_batch_val.shape[0]):
                    if t == 0:
                        I_predict, hidden_states = model(x_batch_val[t], None)
                        loss = loss_fn(I_predict, None, y_batch_val[t], None, rec_fun, first_iteration=True).sum()
                    else:
                        I_predict, hidden_states = model(x_batch_val[t], hidden_states)
                        loss = loss_fn(I_predict, I_predict_previous, y_batch_val[t], y_batch_val[t - 1], rec_fun).sum()
                    # update variables
                    I_predict_previous = I_predict
                    batch_loss += loss.item()
                epoch_losses.append(batch_loss)  # loss for single epoch
            val_losses.append(np.sum(epoch_losses))

    return train_losses, val_losses