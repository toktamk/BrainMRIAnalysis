import os
import cv2
import numpy as np
import random
class MRIDataRead():
    def __init__(self, root,newdim):
        if root == '':
            root = "D://datasets//MRI_Mahdieh_Datasets//task1//"
        self.root = root
        self.dirs = os.listdir(root)
        self.newdim = newdim
        self.channel_num = 1
    def ReadData(self):   
        total_mris = []
        targets = []
        root = self.root
        for dir1 in self.dirs:
            path = root + dir1 + "//"
            dirs2 = os.listdir(path)
            target1 = dir1
            for dir2 in dirs2:
                path2 = path + dir2 + "//"
                files = os.listdir(path2)
                patient_mri = []
                for file1 in files:
                    img = cv2.imread(path2+file1)
                    img = cv2.resize(img,(self.newdim,self.newdim))
                    try:
                        self.channel_num = img.shape[2]
                    except:
                        self.channel_num = 1
                    img = img.reshape(-1)
                    patient_mri.append(img)
                total_mris.append(patient_mri)
                targets.append(target1)
        return total_mris,targets,self.channel_num
    def ChooseBagIndex(self,bagsize,image_num):
        bag_index = []
        bin_size = round(image_num/bagsize)
        for k in range(bagsize):
            random_integer = random.randint((k*bin_size+1), (k+1)*bin_size)
            if random_integer >= image_num:
                random_integer = image_num -1
            bag_index.append(random_integer)
        bag_index = np.array(bag_index)
        return bag_index
        
    def GenBags(self,total_mris,targets,bagsize=3,bags_num=6):
        data_bags = []
        target_bags = []
        t_mri = total_mris
        targets = np.array(targets)
        for i in range(targets.shape[0]):
            patient_mri = np.array(t_mri[i])
            target1 = 0
            target1 = targets[i]
            if target1 == 'grade1':
                target1 = 1
            if target1 == 'grade2':
                target1 = 2
            if target1 == 'grade3':
                target1 = 3
            if target1 == 'grade4':
                target1 = 4
            for k in range(bags_num):
                tmp_bag = []
                bag_index = self.ChooseBagIndex(bagsize,patient_mri.shape[0])
                for p in bag_index:
                    tmp_bag.append(patient_mri[p])
                data_bags.append(tmp_bag)
                target_bags.append(target1)
        return data_bags, target_bags
