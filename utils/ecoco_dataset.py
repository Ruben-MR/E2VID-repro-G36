import os.path

import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
from config import DATA_DIR
from torchvision.io import read_image


class ECOCO_Dataset(torch.utils.data.Dataset):

    def __init__(self, start_index=0, sequence_length=8, path=DATA_DIR):
        """
        Dataset class for instantiating individual elements of the event dataset.

        :param sequence_length: the length of the video sequence. By this we mean the number of transition steps between
        images in the video sequence. This essentially means that the produced sequences will contain
        (sequence_length + 1) images, and the corresponding transition tensors (both flow, and event) in between those
        images (so, an amount equal to sequence_length, each).

        :param start_index: the first index from which to retrieve the sequence.

        :param path: the path of the directory within which the dataset directory can be found.

        As an example, with start_index=8 and sequence_length=3,
        the indices of the tensors to retrieve will be:
            - event: [8, 9, 10]
            - frame: [8, 9, 10, 11]
            - flow: [9, 10, 11]
        (differences in the indices per type of tensor are due to the way the database indexing is done)

        """
        assert start_index <= 50

        self.sequence_length = sequence_length
        self.start_idx = start_index
        self.root_path = path

    def __getitem__(self, idx):
        """
        Method for retrieving the relevant tensors of sequence specified by idx
        :param idx: the index corresponding to the desired sequence to read from.
        :return:
            - events, a torch.Tensor of shape [self.sequence_length, 5, 180, 240]
            - frames, a torch.Tensor of shape [(self.sequence_length + 1), 180, 240]
            - flows, a torch.Tensor of shape [self.sequence_length, 2, 180, 240]
        """
        assert idx <= 1000

        if idx < 950:
            sequence_path = os.path.join(self.root_path,
                                         "ecoco_depthmaps_test",
                                         "train",
                                         "sequence_{:>010d}".format(idx))
        else:
            sequence_path = os.path.join(self.root_path,
                                         "ecoco_depthmaps_test",
                                         "validation",
                                         "sequence_{:>010d}".format(idx))
        # print(f"{sequence_path=}")

        flow_path = os.path.join(sequence_path, "flow")
        frame_path = os.path.join(sequence_path, "frames")
        event_path = os.path.join(sequence_path, "VoxelGrid-betweenframes-5")
        # print(f"{flow_path=}")
        # print(f"{frame_path=}")
        # print(f"{event_path=}")

        frames = torch.zeros(size=[self.sequence_length + 1, 180, 240])
        flows = torch.zeros(size=[self.sequence_length, 2, 180, 240])
        events = torch.zeros(size=[self.sequence_length, 5, 180, 240])

        for i, sequence_index in enumerate(range(self.start_idx, self.start_idx + self.sequence_length)):
            print(f"{i, sequence_index=}")
            flow = np.load(os.path.join(flow_path, "disp01_{:>010d}.npy".format(sequence_index+1)))
            event = np.load(os.path.join(event_path, "event_tensor_{:>010d}.npy".format(sequence_index)))
            frame = read_image(os.path.join(frame_path, "frame_{:>010d}.png".format(sequence_index)))

            # print(f"{type(flow), flow.shape=}")
            # print(f"{type(event), event.shape=}")
            # print(f"{type(frame), frame.shape=}")
            flows[i] = torch.from_numpy(flow)
            events[i] = torch.from_numpy(event)
            frames[i] = frame

        # print(f"{self.sequence_length, self.start_idx + self.sequence_length=}")
        last_frame = read_image(os.path.join(frame_path,
                                             "frame_{:>010d}.png".format(self.start_idx + self.sequence_length)))
        print(f"{type(last_frame), last_frame.shape=}")
        frames[self.sequence_length] = last_frame

        return events, frames, flows

    def __len__(self):
        raise NotImplementedError


if __name__ == '__main__':

    seq_length = 3
    start_idx = 8
    data_path = DATA_DIR

    dataset = ECOCO_Dataset(sequence_length=seq_length, start_index=start_idx, path=data_path)

    # print(f"{dataset.sequence_length=}")
    # print(f"{dataset.start_idx=}")
    # print(f"{dataset.root_path=}")

    event_tensor, frame_tensor, flow_tensor = dataset.__getitem__(42)

    print(f"{type(event_tensor), event_tensor.shape=}")
    print(f"{type(frame_tensor), frame_tensor.shape=}")
    print(f"{type(flow_tensor), flow_tensor.shape=}")

    # print("\n\nFrames")
    # for i, image in enumerate(frame_tensor):
    #     plt.imshow(image)
    #     plt.show()
    #
    # print("\n\nEvents")
    # for i, event in enumerate(event_tensor):
    #     print(event)
    #
    # print("\n\nFlows")
    # for i, flow in enumerate(flow_tensor):
    #     print(flow)

