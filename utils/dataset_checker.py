import os
from config import DATA_DIR
import numpy as np


if __name__ == "__main__":
    event_nums = []
    for i in range(950):
        main_path = os.path.join(DATA_DIR,
                                 "ecoco_depthmaps_test",
                                 "train",
                                 "sequence_{:>010d}".format(i))

        if not os.path.isdir(main_path):
            print("Sequence {} is missing".format(i))
            continue

        flow_path = os.path.join(main_path, "flow")

        files = os.listdir(flow_path)
        prev_file_num = int(files[1][7:-4])
        for j, file in enumerate(files[2:]):
            num = int(file[7:-4])
            if num - prev_file_num != 1:
                print("Sequence {} has no flow tensor number {}".format(i, prev_file_num + 1))
            prev_file_num = num

        num_flows = j + 1
        #print("Sequence {} has {} flow tensors".format(i, j + 1))

        frame_path = os.path.join(main_path, "frames")

        files = os.listdir(frame_path)
        prev_file_num = int(files[0][6:-4])
        for j, file in enumerate(files[1:-2]):
            num = int(file[6:-4])
            if num - prev_file_num != 1:
                print("Sequence {} has no frame number {}".format(i, prev_file_num + 1))
            prev_file_num = num

        #print("Sequence {} has {} images".format(i, j + 1))
        num_images = j + 1

        event_path = os.path.join(main_path, "VoxelGrid-betweenframes-5")

        files = os.listdir(event_path)
        prev_file_num = int(files[1][13:-4])
        for j, file in enumerate(files[2:-2]):
            num = int(file[13:-4])
            if num - prev_file_num != 1:
                print("Sequence {} has no event tensor number {}".format(i, prev_file_num + 1))
            prev_file_num = num

        print("Sequence {} has {} event tensors\n".format(i, j + 1))
        num_events = j + 2
        event_nums.append(num_events)
        #if not (num_events == num_flows and num_events == num_images - 1):
            #print("Sequence {} has an error".format(i))
    event_nums = np.sort(np.array(event_nums))
    print(event_nums[:10])
