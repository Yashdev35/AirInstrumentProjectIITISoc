import numpy as np
import math
import cv2
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
def volControl(img, landmarks):
    if len(landmarks) != 0:
        for hand_landmarks in landmarks:
            # Get the coordinates of the thumb tip (landmark 4) and the index finger tip (landmark 8)
            x1, y1 = int(hand_landmarks.landmark[4].x * img.shape[1]), int(hand_landmarks.landmark[4].y * img.shape[0])
            x2, y2 = int(hand_landmarks.landmark[8].x * img.shape[1]), int(hand_landmarks.landmark[8].y * img.shape[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            # Visual elements on the image for debugging or feedback
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            # Calculate the length between the thumb tip and the index finger tip
            length = math.hypot(x2 - x1, y2 - y1)
            
            # Interpolate volume levels
            vol = np.interp(length, [50, 300], [minVol, maxVol])
            volBar = np.interp(length, [50, 300], [400, 150])
            volPer = np.interp(length, [50, 300], [0, 100])
            
            # Set the system volume
            volume.SetMasterVolumeLevel(vol, None)
            
            # Visual feedback for volume level
            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
            cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
                
    return img