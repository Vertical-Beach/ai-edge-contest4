import cv2
import os
import glob

files = glob.glob("./rst/*.png")
for file in files:
    print(file)
    img = cv2.imread(file)
    img = cv2.resize(img, (1936, 1216), cv2.INTER_NEAREST)
    dstpath = "./rst2/" + os.path.basename(file)
    cv2.imwrite(dstpath, img)