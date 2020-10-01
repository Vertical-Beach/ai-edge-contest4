import os
import glob

paths = glob.glob('./labels/train/*.png')
for path in paths:
    basename = os.path.basename(path)
    print(basename.replace('_train_id.png', ''))