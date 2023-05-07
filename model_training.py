#!/usr/bin/env python
# coding: utf-8

# In[89]:


import cv2
import numpy as np;
import matplotlib.pyplot as plt
from PIL import Image
import imageio  
import skimage
import skimage.color as skic
import skimage.filters as skif
import skimage.data as skid
import skimage.util as sku
#print(Imagethumb.shape)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop
import pywt
import pywt.data
import mediapipe as mp
import uuid
import os
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 
mp_hands = mp.solutions.hands

from PIL import Image
import os
count=1;
Imagethumb=[]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop
import pywt
import pywt.data
import mediapipe as mp
import uuid
import os
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 
mp_hands = mp.solutions.hands

mp_pose=[]
mp_pose = mp.solutions.pose

pose_image=[]
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.001)

mp_model=[]
mp_model = mp_hands.Hands( static_image_mode=True, 
                          max_num_hands=2, 
                          min_detection_confidence=0.001) 

from PIL import Image
import os
count=1;
Imagethumb=[]
    


# In[90]:


NumPres=0;
Seen=0;
Train=[];
x1=[]

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Closed/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;

    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(0);


# In[92]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Closed 2/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;

    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(0);


# In[94]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Down/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape # 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(1);


# In[96]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Down 2/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(1);


# In[98]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Open/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(2);


# In[100]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Open 2/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(2);


# In[102]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Open 3/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(2);


# In[104]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Pointing/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(3);


# In[106]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Pointing 2/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(3);


# In[108]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Pointing 3/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(3);


# In[110]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Pointing 4/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(3);


# In[112]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Up/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(4);


# In[114]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Up 2/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(4);


# In[116]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/As/Up 3/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
 
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Train.append(Array);
        os.remove(img)
        
for i in range(NumPres):
    x1.append(4);


# In[118]:


NumPres=0;
Seen=0;
x2=[]
Test=[]

for img in glob.glob("/Users/anuraagthakur/Desktop/Closed/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Test.append(Array);
        os.remove(img)

for i in range(NumPres):
    x2.append(0);


# In[128]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/Down/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Test.append(Array);
        os.remove(img)

for i in range(NumPres):
    x2.append(1);


# In[122]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/Open/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Test.append(Array);
        os.remove(img)

for i in range(NumPres):
    x2.append(2);


# In[124]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/Pointing/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Test.append(Array);
        os.remove(img)

for i in range(NumPres):
    x2.append(3);


# In[126]:


NumPres=0;
Seen=0;

for img in glob.glob("/Users/anuraagthakur/Desktop/Up/*.png"):
    cv_img=[]
    Y2=[]
    cv_img = cv2.imread(img)
    image=[]
    image = cv_img
    count=count+1;
    results=[]
    results = mp_model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_height=[]
    image_width=[]
    c=[]
    image_height, image_width, c = image.shape 
    thumbwidth=[]
    thumbheight=[]
    Seen=Seen+1;
    
    if results.multi_hand_landmarks is not None:
        NumPres=NumPres+1;
        for hand_landmarks in results.multi_hand_landmarks:
            thumbwidth.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
            thumbheight.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
            
        index_min=[]
        index_min = np.argmin(thumbheight)

        Array=[];
        counter=-1;
        
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
                
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height)


        counter=-1;
        for hand_landmarks in results.multi_hand_landmarks:
            counter=counter+1
            if(index_min==counter):
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width)

                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width)
                Array.append(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width)

        Test.append(Array);
        os.remove(img)

for i in range(NumPres):
    x2.append(4);


# In[131]:


#Do some normalization for the image sent in
import numpy
Array = numpy.zeros((42,1))
#The array to store the final image
Final=[]

#The outer loop goes through the entire image
for i in range(np.array(x1).size):
    #counter for which keypoint we are in
    #increment when first encounter so it is set to -1
    counter=-1;
    
    #The keypoints for the height
    Array2=numpy.zeros((21,1))
    for j in range(21):
        counter=counter+1;
        Array2[j][0]=Train[i][counter] 
        
    Min1= min(Array2)
    
    Minimum1=numpy.zeros((21,1))
    for j in range(21):
        Minimum1[j][0]=Min1     
        
    #The keypoints for the width
    Array3=numpy.zeros((21,1))
    for j in range(21):
        counter=counter+1;
        Array3[j][0]=Train[i][counter]
        
    Min2=min(Array3)

    Minimum2=numpy.zeros((21,1))
    for j in range(21):
        Minimum2[j][0]=Min2
        

    Max1= max(Array2-Minimum1)
    Max2= max(Array3-Minimum2)
    Array = numpy.zeros((42,1))
    
    #Find the maximum and normalize after center algining it by substracting by the minimum 
    Array=np.concatenate([(Array2-Minimum1)/Max1, (Array3-Minimum2)/Max2])
    
    # Save the modified image
    Final.append(Array)
    
    
Final=np.array(Final)


# In[132]:


#Same logic as abvoe but for the test cases
#Center alginment and normalization
import numpy
Array = numpy.zeros((42,1))
Tester=[]


for i in range(np.array(x2).size):
    counter=-1;
    
    Array2=numpy.zeros((21,1))
    for j in range(21):
        counter=counter+1;
        Array2[j][0]=Test[i][counter] 
        
    Min1= min(Array2)
    
    Minimum1=numpy.zeros((21,1))
    for j in range(21):
        Minimum1[j][0]=Min1     
        
    
    Array3=numpy.zeros((21,1))
    for j in range(21):
        counter=counter+1;
        Array3[j][0]=Test[i][counter]
        
    Min2=min(Array3)
    
    Minimum2=numpy.zeros((21,1))
    for j in range(21):
        Minimum2[j][0]=Min2
    
    Max1= max(Array2-Minimum1)
    Max2= max(Array3-Minimum2)
    Array = numpy.zeros((42,1))
    Array=np.concatenate([(Array2-Minimum1)/Max1, (Array3-Minimum2)/Max2])
    
    
    Tester.append(Array)
    
    
Tester=np.array(Tester)


# In[133]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import glob

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import SGD, RMSprop
import pywt
import pywt.data
import mediapipe as mp
import uuid
import os
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles 
mp_hands = mp.solutions.hands


# In[134]:


import math
# Runs all combinaitons of fucntions and Number of nodes possible for a gvien number of layers
def power_of(n,base):
    Array=[]
    Cons=size
    for i in range(Cons):
        c=math.floor(n/pow(base,Cons-1))
        Array.append(c)
        n=n-c*pow(base,Cons-1)
        Cons=Cons-1        
    return(Array)


# In[137]:


import tensorflow as tf
import keras

FunctionSpace=["relu","selu"]
base=2;
ValArray=0;
NumNodes=[50,100];
NodeComb=[]
FuncComb=[]

for numberofLayers in range(2,3,1):
    size=numberofLayers;

    for j in range(pow(np.array(NumNodes).size,(size))):
        H=power_of(j,np.array(NumNodes).size)

        for n in range(pow(base,(size))):
            S=power_of(n,base)  

            model = Sequential()
            model.add(Conv2D(NumNodes[H[0]], (4,1), activation=FunctionSpace[S[0]], kernel_initializer='he_uniform', input_shape=(42,1,1)))            
            model.add(tf.keras.layers.BatchNormalization())
            
            for Numberintlayer in range(1,numberofLayers,1): 
                print(H[Numberintlayer])
                print(S[Numberintlayer])
                model.add(Dense(NumNodes[H[Numberintlayer]], activation=FunctionSpace[S[Numberintlayer]], kernel_initializer='he_uniform'))

            model.add(Flatten())
            model.add(Dense(5, activation='softmax'))
            numberofLayers
            print(H)
            print(S)

            # compile model
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(np.array(Final), np.array(x1), 
                                epochs=10, batch_size=2, 
                                validation_data=(np.array(Tester), np.array(x2)))
            if((max(history.history['val_accuracy']))>ValArray):
                ValArray=max(history.history['val_accuracy'])
                NodeComb=H
                SaveNumLayers=numberofLayers
                FuncComb=S


# In[139]:


print(ValArray)
print(NodeComb)
print(SaveNumLayers)
print(FuncComb)


# In[140]:


model = Sequential()
model.add(Conv2D(50, (4,1), activation='relu', kernel_initializer='he_uniform', input_shape=(42,1,1)))            
#model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))


# compile model
#opt = SGD(lr=0.001, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(np.array(Final), np.array(x1), 
                                epochs=10, batch_size=2, 
                                validation_data=(np.array(Tester), np.array(x2)))


# In[322]:


import pickle

filename = 'uytmodel.sav'
pickle.dump(model, open(filename, 'wb'))
 
base=12;
size=2;


# In[248]:


model.save('TryTOSee')

