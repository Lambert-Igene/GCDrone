from tensorflow import keras
import cv2
import os

import numpy as np
from enum import Enum
import mediapipe as mp

class Gestures(Enum):
    CLOSED=0
    DOWN=1
    OPEN=2
    POINT=3
    UP=4
    NO_CONF=5
    def __str__(self):
        return self.name

class GestureDetector():
    def __init__(self):
        self.model = keras.models.load_model(r"GestureModel")
        self.model.summary()
        self.mp_hands = mp.solutions.hands

    def get_model_input(self,hand_landmarks,image_width,image_height):
        thumbwidth=[]
        thumbheight=[]
        Array=[]
        New=[]
        Newer=[]
        
        thumbwidth.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].x * image_width)
        thumbheight.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].y * image_height)
        
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image_height)
               
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP].y * image_height)

        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)
        
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height)

        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP].y * image_height)
        Array.append(hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP].y * image_height)
        
        New.append(Array)
        Array = np.zeros((42,1))
        
        for i in range(1):
            counter=-1
        
            Array2=np.zeros((21,1))
            for j in range(21):
                counter=counter+1
                Array2[j][0]=New[i][counter]
            
            Min1= min(Array2)
        
            Minimum1=np.zeros((21,1))
            for j in range(21):
                Minimum1[j][0]=Min1    
            
            counter=-1
            Array3=np.zeros((21,1))
            for j in range(21):
                counter=counter+1
                Array3[j][0]=New[i][counter]
            
            Min2=min(Array3)
            Minimum2=np.zeros((21,1))
            for j in range(21):
                Minimum2[j][0]=Min2
        
            Max1= max(Array2-Minimum1)
            Max2= max(Array3-Minimum2)
        
            Array=np.concatenate([(Array2-Minimum1)/Max1, (Array3-Minimum2)/Max2])
        
        
            Newer.append(Array)
        
        Newer=np.array(Newer)

        return Newer

    def getHandGesture(self, image, handLMs, bbox_color=(0, 255, 0)):  
        
        h, w, c = image.shape
        # print(image.shape)
        x_max = 0
        y_max = 0
        x_min = w
        y_min = h
        for lm in handLMs.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            if x > x_max:
                x_max = x
            if x < x_min:
                x_min = x
            if y > y_max:
                y_max = y
            if y < y_min:
                y_min = y
        # drawing the bounding box
        extra = 10
        cv2.rectangle(image, (x_min - extra, y_min - extra), (x_max + extra, y_max + extra), bbox_color, 1)
        image = image[max(y_min - extra, 0):min(y_max + extra, h), max(x_min - extra, 0):min(x_max + extra, w)]
        
        # get normilized mediapipe hand skeleton input for model
        input = self.get_model_input(handLMs,h,w)

        # Predict the gesture
        pred = self.model.predict(input)
        pred_class = np.argmax(pred)
        if np.max(pred) <= 0.5:
            pred_class = 5
        return pred_class
    
