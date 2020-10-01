import cv2
import glob
import os
paths = glob.glob("../seg_train_images/*.jpg")
print(len(paths))
for path in paths:
    img = cv2.imread(path)
    img = cv2.flip(img, 1)
    basename = os.path.basename(path).replace('.jpg', '')
    dstpath = "../seg_train_images/" + basename + "_flip.jpg"
    cv2.imwrite(dstpath, img)
    print(dstpath)

paths = glob.glob("../seg_train_images/*.jpg")
print(len(paths))