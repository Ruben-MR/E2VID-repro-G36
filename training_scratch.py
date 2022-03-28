from utils.ecoco_sequence_loader import *
from model.model import E2VIDRecurrent
from utils.train_utils import PreProcessOptions, RescalerOptions
from utils.inference_utils import EventPreprocessor, IntensityRescaler
from utils.train_utils import plot_training_data, pad_all, loss_fn, training_loop
from utils.loading_utils import get_device

if __name__ == "__main__":
    # ======================================================================================================================================================
    # Model definition
    config = {'recurrent_block_type': 'convlstm', 'num_bins': 5, 'skip_type': 'sum', 'num_encoders': 3,
              'base_num_channels': 32, 'num_residual_blocks': 2, 'norm': 'BN', 'use_upsample_conv': True}
    model = E2VIDRecurrent(config=config).cuda()

    # Event preprocessor
    options = PreProcessOptions()
    preprocessor = EventPreprocessor(options)
    options = RescalerOptions()
    rescaler = IntensityRescaler(options)

    # ignore the code above, they are just used for taking out the event tensor and model
    device = get_device(True)
    height, width = (180, 240)
    # DATA_DIR = '/home/richard/Q3/Deep_Learning/ruben-mr.github.io/data'

    events = torch.tensor(full_event_tensor(range(5), 5, DATA_DIR)[0], dtype=torch.float64).cuda().float()
    images = torch.tensor(full_image_tensor(range(5), 5, DATA_DIR)[0], dtype=torch.float64).cuda().float()

    # =============================
    # data pre-processing
    events, images = pad_all(model, events, images)
    # =============================
    train_loader = [(events, images)]
    validation_loader = [(events, images)]

    train_losses, val_losses = training_loop(model, loss_fn, train_loader, validation_loader)
    plot_training_data(train_losses, val_losses)