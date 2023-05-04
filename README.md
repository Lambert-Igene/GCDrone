# GCDrone
CS 670 Deep Learning Project - Controlling a drone using gestures.

## Installation  
### Only tested using Windows  
#### Download repo.  

#### Create start python environment (Windows Command Line):  
`python3.9 -m venv DroneEnv` 
 
#### Start python environment:  
`DroneEnv\Scripts\activate`

#### Install dependences:
Make sure your pip version is up to date. 
If you are unsure run `python -m pip install --upgrade pip`.  

`pip install -r requirements.txt`  

## Drone Control  
### Connecting to drone:  
Press the drone power button  
Connect to the drone's wifi network on the system you are using to run the code 

### Running Code:
Navigate to the top level repository folder (<Repo Download Location\GCDrone>)  
With all the dependences installed and the RealSense camera connected to your computer run the command:  
`python3 realTimeRealsense.py`  

A window showing the RealSense color camera feed should appear.
Present the start gesture (closed fist) to the camera for about 3 seconds.
Once the start gesture is detected, the drone will take off.
Using the same hand, present thumbs up to move the drone up, thumbs down to move it down, or index finger pointing on a 2d plane to move it that direction.
Show the stop gesture (open hand) to land the drone.

## Dataset
https://drive.google.com/drive/folders/126N4l7qGfchTwiwzxlBmBZYko2HLmrqa?usp=sharing

If you would like to collect more data, run the dataCollection.py file
