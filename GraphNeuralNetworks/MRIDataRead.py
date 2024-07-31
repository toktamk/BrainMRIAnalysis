#MRIDataRead: A module for reading and preprocessing MRI data.

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
    
        
    def GenBags(self,total_mris,targets,bagsize=3):
        data_bags = []
        target_bags = []
        t_mri = total_mris
        
        targets = np.array(targets)
        for i in range(targets.shape[0]):
            patient_mri = np.array(t_mri[i])
            target1 = 0
            target1 = targets[i]
            bag_num = patient_mri.shape[0]-bagsize
            if target1 == 'grade1':
                target1 = 0
            if target1 == 'grade2':
                target1 = 1
            if target1 == 'grade3':
                target1 = 2
            if target1 == 'grade4':
                target1 = 3
            for k in range(bag_num):
                tmp_bag = []
                
                for p in range(bagsize):
                    tmp_bag.append(patient_mri[(k+p)])
                data_bags.append(tmp_bag)
                target_bags.append(target1)
        return data_bags, target_bags
