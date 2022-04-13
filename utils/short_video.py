import glob
import os.path
import cv2
from config import DATA_DIR, SAVED_DIR


def make_video(name):
    filename = os.path.join(DATA_DIR, "results", name)

    print(filename)

    os.chdir(filename)

    video_file = f'{name}.mp4'
    image_size = (240, 180)
    fps = 15

    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, image_size)

    for file in sorted(glob.glob("*.png")):
        print(file)

        img = cv2.imread(file)
        cv2.imshow('', img)
        cv2.waitKey(1)
        out.write(img)

    out.release()


if __name__ == '__main__':

    names = os.listdir(SAVED_DIR)
    names.remove("unused")

    for name in names:
        make_video(name)

