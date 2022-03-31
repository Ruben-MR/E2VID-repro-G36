import os.path
import torch.utils.data
import numpy as np
from config import DATA_DIR
from torchvision.io import read_image
from tqdm import tqdm


class ECOCO_Train_Dataset(torch.utils.data.Dataset):

    N_SEQUENCES = 950

    def __init__(self, start_index=0, sequence_length=8, shift=2, n_shifts=1, path=DATA_DIR):
        """
        Dataset class for instantiating individual elements of the event dataset.

        :param sequence_length: the length of the video sequence. By this we mean the number of transition steps between
        images in the video sequence. This essentially means that the produced sequences will contain
        (sequence_length + 1) images, and the corresponding transition tensors (both flow, and event) in between those
        images (so, an amount equal to sequence_length, each).
        :param start_index: the first index from which to retrieve the sequence.
        :param shift: the amount of time indices to 'shift' by. This means that sequences within the dataset are
        essentially reused multiple times, and each time a sequence is reused, a different time window is being
        extracted. Irrelevant in the case that n_shifts = 1
        :param n_shifts: the total number of shifts considered. With the default value of 1, no multiple shifts will be
        performed, each sequence within the dataset is used only one time.
        :param path: the path of the directory within which the dataset directory can be found.

        As an example, with start_index=8 and sequence_length=3,
        the indices of the tensors to retrieve will be:
            - event: [8, 9, 10]
            - frame: [8, 9, 10, 11]
            - flow: [9, 10, 11]
        (differences in the indices per type of tensor are due to the way the database indexing is done)
        """
        assert n_shifts > 0
        assert start_index + (n_shifts - 1) * shift + sequence_length <= 55

        self.sequence_length = sequence_length
        self.start_idx = start_index
        self.shift = shift
        self.n_shifts = n_shifts
        self.root_path = path

    def __getitem__(self, idx):
        """
        Method for retrieving the relevant tensors of sequence specified by idx
        :param idx: the index corresponding to the desired sequence to read from.
        :return:
            - events, a torch.Tensor of shape [self.sequence_length, 5, 180, 240]
            - frames, a torch.Tensor of shape [(self.sequence_length + 1), 1, 180, 240]
            - flows, a torch.Tensor of shape [self.sequence_length, 2, 180, 240]
        """
        assert idx < self.__len__()

        while idx % self.N_SEQUENCES in [107, 382]:
            idx = int(np.random.randint(0, self.__len__(), (1,)))

        shft_idx, idx = (idx // self.N_SEQUENCES, idx % self.N_SEQUENCES)
        # print(f"{(shft_idx, idx)=}")

        sequence_path = os.path.join(self.root_path,
                                     "ecoco_depthmaps_test",
                                     "train",
                                     "sequence_{:>010d}".format(idx))
        # print(f"{sequence_path=}")

        flow_path = os.path.join(sequence_path, "flow")
        frame_path = os.path.join(sequence_path, "frames")
        event_path = os.path.join(sequence_path, "VoxelGrid-betweenframes-5")
        # print(f"{flow_path=}")
        # print(f"{frame_path=}")
        # print(f"{event_path=}")

        frames = torch.zeros(size=[self.sequence_length + 1, 1, 180, 240])
        flows = torch.zeros(size=[self.sequence_length, 2, 180, 240])
        events = torch.zeros(size=[self.sequence_length, 5, 180, 240])

        for i, sequence_index in enumerate(range(self.start_idx + shft_idx * self.shift,
                                                 self.start_idx + shft_idx * self.shift + self.sequence_length)):
            # print(f"{(i, sequence_index)=}")
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
        # print(f"{type(last_frame), last_frame.shape=}")
        frames[self.sequence_length] = last_frame

        return events.cuda(), frames.cuda(), flows.cuda()

    def __len__(self):
        return self.N_SEQUENCES * self.n_shifts


class ECOCO_Validation_Dataset(torch.utils.data.Dataset):

    N_SEQUENCES = 50

    def __init__(self, start_index=0, sequence_length=8, shift=2, n_shifts=1, path=DATA_DIR):
        """
        Dataset class for instantiating individual elements of the event dataset.

        :param sequence_length: the length of the video sequence. By this we mean the number of transition steps between
        images in the video sequence. This essentially means that the produced sequences will contain
        (sequence_length + 1) images, and the corresponding transition tensors (both flow, and event) in between those
        images (so, an amount equal to sequence_length, each).
        :param shift: the amount of time indices to 'shift' by. This means that sequences within the dataset are
        essentially reused multiple times, and each time a sequence is reused, a different time window is being
        extracted. Irrelevant in the case that n_shifts = 1
        :param n_shifts: the total number of shifts considered. With the default value of 1, no multiple shifts will be
        performed, each sequence within the dataset is used only one time.
        :param start_index: the first index from which to retrieve the sequence.
        :param path: the path of the directory within which the dataset directory can be found.
        """
        assert n_shifts > 1
        assert start_index + (n_shifts - 1) * shift + sequence_length <= 55

        self.sequence_length = sequence_length
        self.start_idx = start_index
        self.shift = shift
        self.n_shifts = n_shifts
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

        assert idx < self.__len__()
        shft_idx, idx = (idx // self.N_SEQUENCES, idx % self.N_SEQUENCES)
        idx += 950
        # print(f"{(shft_idx, idx)=}")

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

        frames = torch.zeros(size=[self.sequence_length + 1, 1, 180, 240])
        flows = torch.zeros(size=[self.sequence_length, 2, 180, 240])
        events = torch.zeros(size=[self.sequence_length, 5, 180, 240])

        for i, sequence_index in enumerate(range(self.start_idx + shft_idx * self.shift,
                                                 self.start_idx + shft_idx * self.shift + self.sequence_length)):
            # print(f"{(i, sequence_index)=}")
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
        # print(f"{type(last_frame), last_frame.shape=}")
        frames[self.sequence_length] = last_frame

        return events.cuda(), frames.cuda(), flows.cuda()

    def __len__(self):
        return self.N_SEQUENCES * self.n_shifts


if __name__ == '__main__':

    seq_length = 8
    shift = 4
    n_shifts = 2
    start_idx = 20
    data_path = DATA_DIR
    sequence_index = 2000

    train_dataset = ECOCO_Train_Dataset(start_index=start_idx, sequence_length=seq_length, shift=shift, n_shifts=n_shifts, path=data_path)
    val_dataset = ECOCO_Validation_Dataset(start_index=start_idx, sequence_length=seq_length, shift=shift, n_shifts=n_shifts, path=data_path)

    #print(f"{dataset.sequence_length=}")
    #print(f"{dataset.start_idx=}")
    #print(f"{dataset.root_path=}")

    # event_tensor, frame_tensor, flow_tensor = train_dataset.__getitem__(sequence_index)
    #
    # print(f"{type(event_tensor), event_tensor.shape=}")
    # print(f"{type(frame_tensor), frame_tensor.shape=}")
    # print(f"{type(flow_tensor), flow_tensor.shape=}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    print("CHECKING TRAIN LOADER")

    for events, frames, flows in train_loader:
        assert events.shape == torch.Size([2, seq_length, 5, 180, 240]) and frames.shape == torch.Size([2, seq_length+1, 1, 180, 240]) \
               and flows.shape == torch.Size([2, seq_length, 2, 180, 240])

    print("CHECKING VALIDATION LOADER")

    for events, frames, flows in val_loader:
        assert events.shape == torch.Size([2, seq_length, 5, 180, 240]) and frames.shape == torch.Size(
            [2, seq_length + 1, 1, 180, 240]) \
               and flows.shape == torch.Size([2, seq_length, 2, 180, 240])

        # print(f"{(events.is_cuda, frames.is_cuda, flows.is_cuda)=}")
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

