import argparse
from utils.timers import Timer
from utils.loading_utils import load_model, get_device
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid_pytorch
from options.inference_options import set_inference_options
from utils.ecoco_sequence_loader import *
from utils.train_utils import plot_training_data, pad_all, loss_fn, training_loop
import lpips
from utils.inference_utils import IntensityRescaler
from image_reconstructor import ImageReconstructor


if __name__ == "__main__":
    # ======================================================================================================================================================
    # Model definition
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
    device = get_device(args.use_gpu)
    model = load_model(args.path_to_model, map_location = device)
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
    device = get_device(True)
    height, width = (180, 240)
    #DATA_DIR = '/home/richard/Q3/Deep_Learning/ruben-mr.github.io/data'

    events = torch.tensor(full_event_tensor(range(10), 5, DATA_DIR)[0], dtype=torch.float64).cuda().float()
    images = torch.tensor(full_image_tensor(range(10), 5, DATA_DIR)[0], dtype=torch.float64).cuda().float()

    #=============================
    # data pre-processing
    events, images = pad_all(model, events, images)
    #=============================
    train_loader = [(events, images)]
    validation_loader = [(events, images)]

    if torch.cuda.is_available():
        reconstruction_loss_fn = lpips.LPIPS(net='vgg').cuda()
    else:
        reconstruction_loss_fn = lpips.LPIPS(net='vgg')

    train_losses, val_losses = training_loop(model, loss_fn, train_loader, validation_loader, reconstruction_loss_fn)
    plot_training_data(train_losses, val_losses)
















    #====================================================
    # # old test code
    # # ignore the code above, they are just used for taking out the event tensor and model
    # # let's make a pseudo-dataset!!
    # device = torch.device('cuda:0')
    # events = event_tensor.unsqueeze(dim=0)
    # # ==========================
    # # pre-processing step here (normalizing and padding)
    # crop = CropParameters(width, height, model.num_encoders)
    # events = crop.pad(events) # (1, 5, 184, 240)
    # # ==========================================
    #
    # #events = events.view((1,*events.shape)) # (1, 1, 5, 184, 240)
    # events = events.unsqueeze(dim=0)
    # sequence_length = events.shape[0]
    # batch_size = events.shape[1]
    # events = events.tile((sequence_length, batch_size, 1, 1, 1)) # (sequence_len, batch_size, channel, H, W)
    # events = events.to(device)
    # labels = torch.rand(events.shape).detach()
    # labels = labels[:, :, 0:1, :, :] # TODO: dealing with multiple channels
    # labels = labels.to(device)
    #
    # train_loader = [(events, labels)]
    # validation_loader = [(events.detach(), labels)]
    # #============================================================================
    #
    # train_losses, val_losses = training_loop(model, loss_fn, train_loader, validation_loader)
    # plot_training_data(train_losses, val_losses)
    #===================================================