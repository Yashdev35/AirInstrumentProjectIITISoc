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
def volControl(img, x, y):
    if len(x) != 0:
        for (x_,y_) in zip(x, y):
            for (x__,y__) in zip(x_, y_):
                if np.sqrt((x__ - 710)**2 + (y__ - 60)**2) < 40:
                    cv2.circle(img, (710, 60), 40, (255,255,255), thickness=-1)
                    x1, y1 = x[0][4], y[0][4]
                    x2, y2 = x[0][8], y[0][8]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(img, (x1, y1), 15, (0,0,0), cv2.FILLED)
                    cv2.circle(img, (x2, y2), 15, (0,0,0), cv2.FILLED)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                    cv2.circle(img, (cx, cy), 15, (0,0,0), cv2.FILLED)
                    length = math.hypot(x2 - x1, y2 - y1)
                    vol = np.interp(length, [50, 300], [minVol, maxVol])
                    volBar = np.interp(length, [50, 300], [400, 150])
                    volPer = np.interp(length, [50, 300], [0, 100])
                    volume.SetMasterVolumeLevel(vol, None)
                    if length < 50:
                        cv2.circle(img, (cx, cy), 15, (0, 0, 0), cv2.FILLED)
                    cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
                    cv2.rectangle(img, (50, int(volBar)), (85, 400), (128, 0, 0), cv2.FILLED)
                    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 3)
                else:
                    continue               
    return img