import numpy as np

path = './seg_train_dat/train_0000.dat'
fp = open(path,'rb')
a = np.fromfile(fp, np.uint8, -1)
fp.close()

a = a.reshape((1216, 1936, 1))
print(a.shape)
import cv2
resize_size_x = 512
resize_size_y = 256
img = cv2.resize(a), (resize_size_x, resize_size_y), interpolation=cv2.INTER_NEAREST)
print(img.shape)

img2 = np.zeros((resize_size_y, resize_size_x, 3))
for y in range(resize_size_y):
    for x in range(resize_size_x):
        if(img[y][x] == 0):
            img2[y][x] = [255, 0, 0]
        elif img[y][x] == 1:
            img2[y][x] = [255, 255, 0]
        elif img[y][x] == 1:
            img2[y][x] = [0, 0, 255]
        elif img[y][x] == 1:
            img2[y][x] = [0, 255, 0]
        else:
            img2[y][x] = [0, 0, 0]
cv2.imwrite(img2, "val.png")