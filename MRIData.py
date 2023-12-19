import os
import cv2
import numpy as np
import torchvision
class MRIData(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(MRIData, self).__init__(root, transform=None, target_transform=target_transform)
        if root == '':
            root = "D://datasets//MRI_Mahdieh_Datasets//task1//"
        dirs = os.listdir(root)
        mriPair = []
        targets = []
        for dir1 in dirs:
            path = root + dir1 + "//"
            dirs2 = os.listdir(path)
            target1 = dir1
            for dir2 in dirs2:
                path2 = path + dir2 + "//"
                files = os.listdir(path2)
                mritmp = []
                oldimg = cv2.imread(path2+files[0])
                oldimg = cv2.resize(oldimg,(224,224))
                oldimg = oldimg.reshape(-1)
                for file1 in files[1:]:
                    img = cv2.imread(path2+file1)
                    img = cv2.resize(img,(224,224))
                    img = img.reshape(-1)
                    mriPair.append([oldimg,img])
                    oldimg = img
                    targets.append(target1)
        self.data = mriPair
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_pair = self.data[index]
        if self.transform is not None:
            image_pair = tuple(self.transform(img) for img in image_pair)
        return image_pair


