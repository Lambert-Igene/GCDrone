# -*- coding: utf-8 -*-
"""Xxxx2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13SQcIWXf55DxtXdys8FQzv6Bd1s71voh

##### Copyright 2022 The MediaPipe Authors. All Rights Reserved.
"""

#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""# Hand Landmarks Detection with MediaPipe Tasks

This notebook shows you how to use MediaPipe Tasks Python API to detect hand landmarks from images.

## Preparation

Let's start with installing MediaPipe.

*Notes:*
* *If you see an error about `flatbuffers` incompatibility, it's fine to ignore it. MediaPipe requires a newer version of flatbuffers (v2), which is incompatible with the older version of Tensorflow (v2.9) currently preinstalled on Colab.*
* *If you install MediaPipe outside of Colab, you only need to run `pip install mediapipe`. It isn't necessary to explicitly install `flatbuffers`.*
"""

!pip install -q flatbuffers==2.0.0
!pip install -q mediapipe==0.9.1

"""Then download an off-the-shelf model bundle. Check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) for more information about this model bundle."""

!wget -q https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task

"""## Visualization utilities"""

#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image

"""## Download test image

Let's grab a test image that we'll use later. The image is from [Unsplash](https://unsplash.com/photos/mt2fyrdXxzk).
"""

!wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg

import cv2
from google.colab.patches import cv2_imshow

img = cv2.imread("image.jpg")
cv2_imshow(img)
# image.jpg

"""Optionally, you can upload your own image. If you want to do so, uncomment and run the cell below."""

from google.colab import drive
# Mounting my Google drive
drive.mount('/content/drive')

import os
import glob
import cv2


#Setting the local folder path
# os.getcwd()
# os.chdir(r"/content/drive/MyDrive/data collection/Final_Dataset_Embedded/Open")
files_closed = glob.glob(r"/content/drive/MyDrive/data collection/Final_Dataset/Closed/*.png")
save_dir = "/content/drive/MyDrive/data collection/Final_Dataset_Embedded/Closed"

i = 0 
for img_dir in files_closed:

  # if i == 10:
  #   break
  # else:
  # img = cv2.imread(img_dir)
  from PIL import Image
  import mediapipe as mp
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision

  # STEP 2: Create an ImageClassifier object.
  base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
  options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)

  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(img_dir)

  # STEP 4: Detect hand landmarks from the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the classification result. In this case, visualize it.
  annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

  new_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
  # cv2_imshow(new_image)

  image_name = "/content/drive/MyDrive/data collection/Final_Dataset_Embedded/Closed/Closed%s.png" % str(i)
  cv2.imwrite(image_name, new_image)

  # new_image = Image.fromarray(new_image)
  # new_image.save(image_name)
  # cv2_imshow(new_image)
  
  i +=1 

  # cv2_imshow(img)

# files_closed[0]

type(new_image)

# for i in range(10):
#   print()

image_name = "/content/drive/MyDrive/data collection/new_dataset/Pointing/Pointing%s.jpg" % str(i)
image_name

from google.colab import files
uploaded = files.upload()

for filename in uploaded:
  content = uploaded[filename]
  with open(filename, 'wb') as f:
    f.write(content)

if len(uploaded.keys()):
  IMAGE_FILE = next(iter(uploaded))
  print('Uploaded file:', IMAGE_FILE)

"""## Running inference and visualizing the results

Here are the steps to run hand landmark detection using MediaPipe.

Check out the [MediaPipe documentation](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python) to learn more about configuration options that this solution supports.

"""

# from IPython.core.display import Image
hey = cv2.imread(IMAGE_FILE)
# print(IMAGE_FILE)
# cv2_imshow(hey)
img = cv2.imread(IMAGE_FILE)
len(uploaded)

print(IMAGE_FILE)

len(files)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(files_closed[0])

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))