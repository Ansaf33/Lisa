#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import subprocess


# In[10]:


#BETTER TO HAVE WORKING DIRECTORY IN THE N A Y FILE ITSELF, SINCE WE'LL BE HAVING LOTS OF NIFTI->DICOM CONVERSION


wd = ''
os.chdir(wd)
child_dirs = ['DICOM Files','DICOM Groups','NIFTI Files','Model Details']
il = ['Images','Labels']
titlvivl = ['train_images','train_labels','val_images','val_labels']

for child in child_dirs:
    subprocess.run(['mkdir','-p',"File/"+child])
    
for child in child_dirs[:2]:
    for f in il:
        subprocess.run(['mkdir','-p','File/'+child+'/'+f])
    
for child in titlvivl:
    subprocess.run(['mkdir','-p','File/NIFTI Files/'+child])


# In[ ]:




