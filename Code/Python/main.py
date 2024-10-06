#!/usr/bin/env python
# coding: utf-8

# In[1]:


# FOR PREPARATION
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from glob import glob
import dicom2nifti
import dicom2nifti.settings as settings
settings. disable_validate_slice_increment()

# FOR PREPROCESSING
from preprocessing import show_image,prepare

# FOR TRAINING
import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from fit_implementation import train




# # DATA PREPARATION

# ## STEP 1 : Convert The Dicom Files (Manually Created) into Dicom Groups, each group consisting of SLICES number of slices
# 
# 

# In[2]:


'''

input_image_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Files/Images'
input_label_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Files/Labels'
output_image_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Groups/Images'
output_label_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Groups/Labels'

SLICES = 64

# MOVING THE IMAGES FROM DICOM FILE -> DICOM GROUP

for patient in glob(input_image_path + '/*'):
    patient_name = os.path.basename(os.path.normpath(patient))
    
    number_of_files = int(len(glob(patient+'/*')))

    number_of_folders = int(number_of_files / SLICES)
    
    print(number_of_folders)
    
    for i in range(number_of_folders):
        folder_path = os.path.join(output_image_path, patient_name + '_' + str(i) )
        os.mkdir(folder_path)
        
        for i,file in enumerate(glob(patient + '/*') ):
            
            if i == SLICES + 1:
                break
            shutil.copy(file,folder_path)
        
        
# MOVING THE LABELS FROM DICOM FILE -> DICOM GROUP



for patient in glob(input_label_path + '/*'):
    patient_name = os.path.basename(os.path.normpath(patient))
    
    number_of_files = int(len(glob(patient+'/*')))

    number_of_folders = int(number_of_files / SLICES)
    
    print(number_of_folders)
    
    for i in range(number_of_folders):
        folder_path = os.path.join(output_label_path, patient_name + '_' + str(i) )
        os.mkdir(folder_path)
        
        for i,file in enumerate(glob(patient + '/*') ):
            
            if i == SLICES + 1:
                break
            shutil.copy(file,folder_path)
 
'''
print()


# ## STEP 2 : Convert the DICOM groups each into NIFTI files for both the images and the labels

# In[3]:


'''
input_image_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Groups/Images'
input_label_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/DICOM Groups/Labels'
output_image_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/NIFTI Files/Images'
output_label_path = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/NIFTI Files/Labels'


for file in glob(input_image_path + '/*'):
    patient_name = os.path.basename(os.path.normpath(file))
    dicom2nifti.dicom_series_to_nifti(file,os.path.join(output_image_path, patient_name + ".nii.gz"))



for file in glob(input_label_path + '/*'):
    patient_name = os.path.basename(os.path.normpath(file))
    dicom2nifti.dicom_series_to_nifti(file,os.path.join(output_label_path, patient_name + ".nii.gz"))
'''
print()


# # DATA PREPROCESSING

# ## Call the Prepare function from the 'preprocessing.ipynb' function to return loaders for both the training an validation datasets.

# In[4]:


data_dir = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/NIFTI Files'
model_dir = '/Users/ansafhassan/Documents/Semester V/Image Processing/Group Project/Liver Segmentation/Model Details'
data_input = prepare(data_dir,spatial_size=[128,128,64])
show_image(data_input[0],61)



# # TRAINING THE MODEL

# In[7]:


# CREATE THE MODEL
epoch = 30

model = UNet(
    spatial_dims = 3,
    in_channels = 1,
    out_channels = 2,
    channels = (16,32,64,128,256),
    strides = (2,2,2,2),
    num_res_units = 2,
    norm = Norm.BATCH
)


# CREATE THE LOSS FUNCTION AND OPTIMIZER
loss_function = DiceLoss(sigmoid=True,to_onehot_y=True,squared_pred=True)
optimizer_function = torch.optim.Adam(model.parameters(),1e-5,weight_decay=1e-5,amsgrad=True)


train(model,data_input,loss_function,optimizer_function,20,model_dir,1,torch.device("cpu"))


# # PLOT GRAPH OF LOSS

# In[17]:


train_loss = np.load(os.path.join(model_dir,'training_loss.npy'))
test_loss = np.load(os.path.join(model_dir,'validation_loss.npy'))
train_metric = np.load(os.path.join(model_dir,'training_metric.npy'))
test_metric = np.load(os.path.join(model_dir,'validation_metric.npy'))

x = [x for x in range(epoch)]
y = train_loss
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.plot(x,y)

x = [x for x in range(epoch)]
y = train_metric
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy")
plt.plot(x,y)

x = [x for x in range(epoch)]
y = train_loss
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.plot(x,y)

x = [x for x in range(epoch)]
y = train_metric
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.plot(x,y)




# In[ ]:




