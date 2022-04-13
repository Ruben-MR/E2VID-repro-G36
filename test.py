import numpy as np
import os
from datetime import datetime
from config import LOG_DIR


if __name__ == "__main__":
    data = np.random.randint(0, 1000, (60, 2))
    name = datetime.now().strftime("saved_%d-%m-%Y_%H-%M.csv")
    fullpath = os.path.join(LOG_DIR, name)
    np.savetxt(fullpath, data, delimiter=',')

    """
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

