# Setup streamlit app file
import os
import cv2
import uuid
import torch
import glob
from tqdm import tqdm
import streamlit as st

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from SiameseNetwork import SiameseNetwork
from SiameseData import *

# Making the model run on the GPU
# Load Model
SNN = torch.load("SNN")
SNN.to(torch.device('cuda'))

# Resize the images into 100x100 and transform to tensors
transformation = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

# A method to check the if the model has found a similar person or not with threshold of 0.5 
def state(arr):
    for result in arr:
        if result < 0.5:
            return True

def verify():
  # Loading the application dataset
  folder_dataset_test = datasets.ImageFolder(root=r"application_data\input_image")
  applicationData = app_Data(imageFolderDataset=folder_dataset_test, transform=transformation)
  dataloader = DataLoader(applicationData, num_workers=2, batch_size=1, shuffle=True)
  # Making the predictions and return array of the results the model has done
  results = SNN.predict(dataloader)
  
  # Now check if the model has found similarity or not
  if state(results):
    return "Matches"
  else:
    return "Does not match"

st.title("Face Verification System")

menu = ["Verification", "Add my face"]

choice = st.sidebar.selectbox("Menu", menu)

def Verification():

    FRAME_WINDOW = st.image([])
    
    # if choice == "Verification":
    v = st.button("Verify")
    
    camera = cv2.VideoCapture(0)
    st.markdown("Developed by: Seif_Elkerdany© 2024", unsafe_allow_html=True)
    
    while choice == "Verification":
        ret, frame = camera.read()
        final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(final_frame)
        # Verification trigger
        if v:
            # Save the captured image of the user
            cv2.imwrite('application_data/input_image/input_image.jpg', frame)
            SiameseNetworkDataset.save_face('application_data/input_image/input_image.jpg', 'application_data/input_image')
            status = verify()
            st.success(status)
            v = False
        
def add_face():
    FRAME_WINDOW = st.image([])
    
    add =  st.button("Capture")
    save = st.button("Save")
    
    images = glob.glob(os.path.join(r"VerificationData", '*.jpg'))
    
    cap = cv2.VideoCapture(0)
    st.markdown("Developed by: Seif_Elkerdany© 2024", unsafe_allow_html=True)
    while choice == "Add my face":
        ret, frame = cap.read()

        # Collect images for verification folder
        if add:
            # Save images
            # Note: The more images you add the more time the model will take to predict due to the loop that iterate through all images in the folder
            imgname = os.path.join(r"VerificationData", '{}.jpg'.format(uuid.uuid1()))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (100, 100))
            cv2.imwrite(imgname, img)
            
            st.success("You can capture new image now!")
            add = False
            
        if save:
            for img in tqdm(images):
                if os.path.isfile(os.path.join(r"VerificationData\faces", os.path.basename(img))):
                    pass
                else:
                    try:
                        SiameseNetworkDataset.save_face(img, r"VerificationData\faces")
                    except:
                        pass
            
            # And also for the verification folder 
            for file_name in tqdm(os.listdir(os.path.join(r"VerificationData\faces"))):
                img_path = os.path.join(r"VerificationData\faces", file_name)
                img = cv2.imread(img_path)
                augmented_images = SiameseNetworkDataset.data_aug(img)

                for image in augmented_images:
                    cv2.imwrite(os.path.join(r"application_data\input_image\verification_images", '{}.jpg'.format(uuid.uuid1())), image.numpy())
                    
            st.success("All saved, you can use the application now!")
            save = False

        final_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(final_frame)
    
if __name__ == '__main__':
    
    if choice == "Verification":
        try:
            Verification()
        except:
            st.success("We ran into a problem. \nPlease, try again!")
            
    elif choice == "Add my face":
        try:
            add_face()
        except:
            st.success("We ran into a problem. \nPlease, try again!")