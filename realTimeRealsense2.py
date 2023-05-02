import sys
import mediapipe as mp
import cv2
from cameraStream import cameraStream
import numpy as np

# from tello_dummy import Tello
from djitellopy import Tello
from Gestures import Gestures, GestureDetector
from Director import DirectionDetector

import os
if os.name == 'nt':
    from Timer import WindowsTimer as Timer
elif os.name == 'posix':
    from Timer import LinuxTimer as Timer
else:
    print('Unknown OS')
    exit(0)

hands = mp.solutions.hands.Hands()
gest_detect = GestureDetector()
direction_detect = DirectionDetector()

# tello = Tello()
# tello.connect()

#cap = cv2.VideoCapture(0)
stream = cameraStream(640,480)

write_video = False

writer = None
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


# Predicted class for open/close
pred_class = Gestures.NO_CONF

# Which hand controls start/stop
ControlHand = None

# Buffers for containing delta times for when either hand is OPEN
timer = Timer()
GestSeries = {'Right': [], 'Left': [], 'Total': []}
time_buffer_size = timer.one_second * 5

def get_Direction(hand_landmarks, depth_image):
    mp_hands = mp.solutions.hands

    knuckle_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * 640
    knuckle_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * 480
    knuckle_z = depth_image[int(knuckle_y), int(knuckle_x)]
    tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * 640
    tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * 480
    tip_z = depth_image[int(tip_y), int(tip_x)]

    knuckle_x = np.array(knuckle_x)
    knuckle_x = knuckle_x.astype(np.float32)
    tip_x = np.array(tip_x)
    tip_x = tip_x.astype(np.float32)
    knuckle_z = knuckle_z.astype(np.float32)
    tip_z = tip_z.astype(np.float32)

    # if depth value is 0, find nearest non-zero pixel 
    if(knuckle_z == 0):
        pixels = np.array(np.where(depth_image != 0)).T
        manhatt_dist = np.abs(knuckle_x - pixels[:,0]) + np.abs(knuckle_y - pixels[:,1])
        closest_pixel = pixels[np.argmin(manhatt_dist),:]
        knuckle_z = depth_image[closest_pixel[0], closest_pixel[1]]
        circle = cv2.circle(original_image, closest_pixel, 3, (255, 0, 0), 2)
    if(tip_z == 0):
        pixels = np.array(np.where(depth_image != 0)).T
        manhatt_dist = np.abs(tip_x - pixels[:,0]) + np.abs(tip_y - pixels[:,1])
        closest_pixel = pixels[np.argmin(manhatt_dist),:]
        tip_z = depth_image[closest_pixel[0], closest_pixel[1]]
        circle = cv2.circle(original_image, closest_pixel, 3, (255, 0, 0), 2)

    # hard coded K matrix values
    foc_x = 616.755
    foc_y = 616.829
    pp_x = 316.202
    pp_y = 244.251

    #calculate 3D points using values from K matrix and x and y values
    knuckle_3D_x = (knuckle_x - pp_x) * knuckle_z / foc_x
    knuckle_3D_z = knuckle_z

    tip_3D_x = (tip_x - pp_x) * tip_z / foc_x
    tip_3D_z = tip_z

    # normalize x and z point to a unit vector
    knuckle_vec = np.array((knuckle_3D_x, knuckle_3D_z))
    tip_vec = np.array((tip_3D_x, tip_3D_z))
    dist = knuckle_vec - tip_vec
    norm_dist = (knuckle_vec - tip_vec) / np.linalg.norm(dist)

    # if distance is too big for finger, ignore and move on
    if(np.linalg.norm(dist) > 205):
        last_command = "Incorrect Depth Values - Try Again"
        return last_command

    # scale back up to coord to send to drone 
    max_move = 55
    x_move = norm_dist[0] * max_move
    z_move = norm_dist[1] * max_move

    if((x_move < 0 and x_move > -20) or (x_move > 0 and x_move < 20)):
        # move only in z
        x_move = 0
        z_move = 55 * (z_move/abs(z_move))
    elif((z_move < 0 and z_move > -20) or (z_move < 0)): 
        # move only in x
        x_move = 55 * (x_move/abs(x_move))
        z_move = 0
    
    last_command = ""
    # move left/right (if applicable)
    if(x_move < 0):
        # tello.move_left(abs(int(x_move)))
        print("move left") 
        last_command += "left " + str(abs(int(x_move)))
        print(x_move)
    elif(x_move > 0):
        # tello.move_right(int(x_move))
        print("move right") 
        last_command += "right " + str(abs(int(x_move)))
        print(x_move)
    # move forward/backward (if applicable)
    if(z_move > 0):
        # tello.move_forward(int(z_move))
        print("move forward") 
        last_command += "forward " + str(abs(int(z_move)))
        print(z_move)
    elif(z_move < 0):
        # tello.move_backward(abs(int(z_move)))
        print("move backward") 
        last_command += "backward " + str(abs(int(z_move)))
        print(z_move)
    return last_command



#while cap.isOpened() and ControlHand is None:
try:
    for rgb_image, original_image, _ in stream:
        if ControlHand is not None:
            break
        else:    
            image = np.copy(original_image)
            h, w, c = original_image.shape
            # new_w = 320
            # aspect = h/w
            # original_image = cv2.resize(original_image, (round(new_w), round(new_w*aspect)))
            if write_video and writer is None:
                writer = cv2.VideoWriter('Processed.mp4', fourcc, 30, (640,480))

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h,w,c = image.shape
            result = hands.process(image)
            hand_landmarks = result.multi_hand_landmarks
            last_gests = []
            if hand_landmarks:
                ## TODO Change to do this one time per image (as opposed to one time per hand)
                for handIdx, handLMs in enumerate(hand_landmarks):
                    hand_class = result.multi_handedness[handIdx].classification[0]
                    bbox_color = (255, 0, 0) if hand_class.label == 'Right' else (0, 255, 0)
                    last_gests += [(gest_detect.getHandGesture(original_image, handLMs, bbox_color=bbox_color), hand_class.label)]

                CurGest = {'Right': Gestures.NO_CONF, 'Left': Gestures.NO_CONF}
                # print(last_gests)
                for gest in last_gests:
                    # print(gest[0])
                    CurGest[gest[1]] = Gestures(gest[0])
                    color = (255, 0, 0) if gest[1] == 'Right' else (0, 255, 0)
                    pos = (0, h-10) if gest[1] == 'Right' else (0, 60)
                    cv2.putText(original_image, str(Gestures(gest[0])), pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)

                time_between_frames = timer.GetDelta()
                GestSeries['Right'] += [time_between_frames if CurGest['Right'] == Gestures.CLOSED else 0.0]
                GestSeries['Left'] +=  [time_between_frames if CurGest['Left'] == Gestures.CLOSED else 0.0]
                GestSeries['Total'] += [time_between_frames]

            ## Maintain a buffer of the last X second of gesture time deltas
            crop_indx = 0
            while sum(GestSeries['Total'][crop_indx:]) > time_buffer_size:
                crop_indx += 1
            GestSeries['Right'] = GestSeries['Right'][crop_indx:]
            GestSeries['Left']  = GestSeries['Left'][crop_indx:]
            GestSeries['Total'] = GestSeries['Total'][crop_indx:]

            ## Sum over time-frames
            RightClosed = sum(GestSeries['Right'])
            LeftClosed = sum(GestSeries['Left'])
            TotalTime = max(sum(GestSeries['Total']), time_buffer_size)
            if  RightClosed > (TotalTime / 2) and LeftClosed < (TotalTime / 4):
                ControlHand = 'Right'
            elif LeftClosed > (TotalTime / 2) and RightClosed < (TotalTime / 4):
                ControlHand = 'Left'

            cv2.imshow("Hand",original_image)
            if write_video and writer is not None:
                writer.write(original_image)
            cv2.waitKey(1)
except KeyboardInterrupt:
    stream.kill()
    sys.exit(0)

print('ControlHand is {}'.format(ControlHand))

frame_count = 0

last_command = 'continue'
LastGesture = Gestures.NO_CONF
timer.Restart()
time_between_direction_updates = timer.one_second*2
last_command_time = time_between_direction_updates
# tello.connect()
# tello.takeoff()
# tello.send_command("takeoff")
prev_gestures = []
try:
    for rgb_image, original_image, depth_image in stream:
        h, w, c = original_image.shape
        # new_w = 320
        # aspect = h / w
        # original_image = cv2.resize(original_image, (round(new_w), round(new_w * aspect)))

        image = np.copy(original_image)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h,w,c = image.shape
        result = hands.process(image)
        hand_landmarks = result.multi_hand_landmarks
        telloCommand = None
        if hand_landmarks:
            hand_class = result.multi_handedness[0].classification[0]
            if hand_class.label == ControlHand:

                color = (255, 0, 0) if ControlHand == 'Right' else (0, 255, 0)
                pred_class = gest_detect.getHandGesture(original_image, hand_landmarks[0], bbox_color=color)
                pos = (0,h-10)
                cv2.putText(original_image,str(Gestures(pred_class)), pos, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 5)
                prev_gestures.append(Gestures(pred_class))

                if Gestures(pred_class) == Gestures.POINT:
                    last_hand_land = hand_landmarks[0]
                    last_point_depth = depth_image

            # if direction_idx < len(hand_landmarks):
            #     handLMs = hand_landmarks[direction_idx]
            #     direction_detect.drawOverHands(original_image, handLMs)
            #     telloCommand = direction_detect.GetCommand(handLMs,w,h, depth_image)

        last_command_time += timer.GetDelta()
        if last_command_time > time_between_direction_updates and len(prev_gestures) > 30:
            
            # get the gesture with the most occurance in the last time before updates (default 2 seconds)
            LastGesture = max(prev_gestures,key=prev_gestures.count)
            
            # send command to tello drone based on gesture
            if LastGesture == Gestures.UP:
                last_command = 'up 20'
                # tello.move_up(40)
            elif LastGesture == Gestures.DOWN:
                last_command = 'down 20'
                # tello.move_down(40)
            elif LastGesture == Gestures.POINT:
                # last_command = 'continue'
                last_command = get_Direction(last_hand_land, last_point_depth)
                # last_command = 'go 20 20 20'
            elif LastGesture == Gestures.OPEN:
                last_command = 'land'
                # tello.send_command(last_command)
                # tello.land()
                raise KeyboardInterrupt
            elif LastGesture == Gestures.NO_CONF:
                last_command = 'continue'
                
            # tello.send_command(last_command)
            last_command_time = 0
            prev_gestures = []
        else:
            pass
            # tello.send_command('continue')
            # prev_gestures = []

        cv2.putText(original_image, last_command, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 4)
        frame_count += 1
        cv2.imshow("Hand",original_image)
        if write_video and writer is not None:
            writer.write(original_image)
        # Check for escape key press to quit
        if (cv2.waitKey(1) & 0xFF) == 27:
            break
except KeyboardInterrupt:
    # tello.send_command('land')
    stream.kill()
    if write_video and writer is not None:
        writer.release()

    # tello.plot()
    sys.exit(0)



