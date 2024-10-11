import numpy as np
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset
import torch
import tensorflow as tf
import os
from mtcnn.mtcnn import MTCNN
    
class SiameseNetworkDataset(Dataset):
    """
    Class that inherit from Dataset to get the training and testing images totally at random.
    """
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
        
    def data_aug(img):
        """Method to generate new images with different brightness, quality
        and orientation.

        Args:
            img (array): the image I want to augument

        Returns:
            List: list contains all of the images the mothod created
        """
        data = []
        for i in range(9):
            img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
            img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
            img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
            data.append(img)

        return data
    
    def save_face(img_path, dest_folder):
        """Method to detect and extract faces from the images and save it in another folder

        Args:
            img_path (String): the path of image I want to extract face from it
            dest_folder (String): the path of the folder I want to save the image in it
        """
        # Using MTCNN model to extract faces from images
        img = cv2.imread(img_path)
        detector = MTCNN()
        faces = detector.detect_faces(img)

        # Fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
        x1,y1,w,h = faces[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2 = abs(x1+w)
        y2 = abs(y1+h)

        # Locate the co-ordinates of face in the image
        store_face = img[y1:y2,x1:x2]
        store_face = cv2.resize(store_face, (100, 100))
        cv2.imwrite(os.path.join(dest_folder, os.path.basename(img_path)), store_face)
    
class app_Data(Dataset):
    """
    Class that inherit from Dataset to get the images of the application to make predictions
    """
    
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        
    def __getitem__(self, index):    
        
        img0 = Image.open(r"application_data\input_image\input_image.jpg")
        img0 = img0.convert("L")
        img0 = self.transform(img0)
        
        img1 = Image.open(self.imageFolderDataset.imgs[index][0])
        img1 = img1.convert("L")
        img1 = self.transform(img1) 
        
        return img0, img1
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)
