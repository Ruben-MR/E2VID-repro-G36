from model.model import E2VIDRecurrent
from utils.inference_utils import EventPreprocessor

config = {'recurrent_block_type':'convlstm', 'num_bins':5, 'skip_type':'sum', 'num_encoders':3, 'base_num_channels':32,
          'num_residual_blocks':2, 'norm':'BN', 'use_upsample_conv':True}
model = E2VIDRecurrent(config=config)

class Options:
    def __init__(self):
        self.no_normalize = False
        self.hot_pixels_file = None
        self.flip = False

options = Options()
preprocessor = EventPreprocessor(options)
