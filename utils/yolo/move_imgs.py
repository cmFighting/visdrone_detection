import shutil
import os
import numpy as np
import time


def move_imgs(img_folder, txt_file, dst_folder):
    data = np.loadtxt(txt_file, dtype=str)
    for img_name in data:
        src_img_name = img_name + ".jpg"
        src_img_path = os.path.join(img_folder, src_img_name)
        shutil.copy(src_img_path, dst_folder)
        print("{} copy done!".format(src_img_name))


def timeTest():
    start = time.time()
    print("Start: " + str(start))
    for i in range(1, 100000000):
        pass
    stop = time.time()
    print("Stop: " + str(stop))
    print(str(stop - start) + "秒")


if __name__ == '__main__':
    # move_imgs("E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/JPEGImages", "../txts/test.txt", "E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/Images_split/test")
    # move_imgs("E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/JPEGImages", "../txts/val.txt", "E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/Images_split/val")
    # move_imgs("E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/JPEGImages", "../txts/train.txt", "E:/datas/ai/voc/VisDrone_ROOT/VisDrone2019/Images_split/train")
    timeTest()
