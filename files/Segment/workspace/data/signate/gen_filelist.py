import os
import glob

files = glob.glob("./seg_train_images/*.jpg")
files = sorted(list(files))
for file in files:
    basename = os.path.basename(file).replace('.jpg', '')
    print(basename)