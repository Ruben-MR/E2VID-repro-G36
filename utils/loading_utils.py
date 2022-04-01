import torch
from model.model import *

#==========================================================================
# original load_model()
# def load_model(path_to_model, map_location):
#     print('Loading model {}...'.format(path_to_model))
#     raw_model = torch.load(path_to_model, map_location=map_location)
#     model = E2VIDRecurrent(config=config).cuda()
#     arch = raw_model['arch']
#
#     try:
#         model_type = raw_model['model']
#     except KeyError:
#         model_type = raw_model['config']['model']
#
#     # instantiate model
#     model = eval(arch)(model_type)
#
#     # load model weights
#     model.load_state_dict(raw_model['state_dict'])
#
#     return model
#===========================================================================


def load_model(path_to_model, map_location):
    print('Loading model {}...'.format(path_to_model))
    raw_model = torch.load(path_to_model, map_location=map_location)
    config = {'recurrent_block_type': 'convlstm', 'num_bins': 5, 'skip_type': 'sum', 'num_encoders': 3,
              'base_num_channels': 32, 'num_residual_blocks': 2, 'norm': 'BN', 'use_upsample_conv': True}
    model = E2VIDRecurrent(config=config)#.cuda()

    # load model weights
    model.load_state_dict(raw_model)

    return model


def get_device(use_gpu):
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    return device
