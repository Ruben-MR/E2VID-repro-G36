import glob
import os.path
import cv2
from config import DATA_DIR

name = "saved_07-04-2022_09-05"
filename = os.path.join(DATA_DIR, "results", name)

print(filename)
#
# with open(filename, 'r+') as fp:
#     lines = fp.readlines()
#     indices = [0]
#     indices.extend(list(range(50000, 10050000)))
#     lines = [lines[i] for i in indices]
#
#     print("lines done")
#
#     fp.seek(0)
#     fp.truncate()
#     fp.writelines(lines)
#
#     print(f"truncating done")

# img *= 255
#
#
# cv2.imshow("", img)
# cv2.waitKey()

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
