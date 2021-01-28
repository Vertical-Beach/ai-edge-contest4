import os
import glob

kind = 'test'
paths = glob.glob(f'./images/{kind}/*.jpg')
for path in paths:
    basename = os.path.basename(path)
    print(basename.replace('.jpg', ''))