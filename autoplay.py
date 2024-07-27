import time
import cv2
import numpy as np

from piano_sound import play_piano_sound
from keyboard import Piano
River = ['A3', 'E4', 'F4', 'A4', 'C5', 'E5', 'F5', 'A5', 'C6', 'E6', 'F6', 'A6', 'C7', 'E7']
Fur = ['E4', 'D4', 'E4', 'D4', 'E4', 'B3', 'D4', 'C4', 'A3', 'E3', 'A3', 'B3', 'E3', 'G3', 'B3']
Clair = ['D4', 'A4', 'B4', 'D5', 'G5', 'A5', 'B5', 'D6', 'A6', 'B6', 'D7', 'G7', 'A7', 'B7']
def sound_to_play(notes,white,black,white_piano_notes,black_piano_notes):
    pressed_notes=[]
    keys_to_check=notes
    pressed_keys={"White":[],"Black":[]}
    for key_to_check in keys_to_check:
        flag=False
        for i,key in enumerate(black):
            if key_to_check == black_piano_notes[i]:
                pressed_keys['Black'].append(i)
                pressed_notes.append(black_piano_notes[i])
                break
        if flag:
            continue    
        for i,key in enumerate(white):
            if key_to_check == white_piano_notes[i]:
                pressed_keys['White'].append(i)
                pressed_notes.append(white_piano_notes[i])
                break
        pressed_notes=list(set(pressed_notes))
        pressed_keys['White']=list(set(pressed_keys['White']))
        pressed_keys['Black']=list(set(pressed_keys['Black']))
    
    return pressed_keys,pressed_notes
        
def auto_play(x, y, img, white,black,white_piano_notes,black_piano_notes):
    pressed_notes=[]
    pressed_keys={"White":[],"Black":[]}
    inside = lambda p, c1, c2: c1[0] <= p[0] <= c2[0] and c1[1] <= p[1] <= c2[1] if c1[0] < c2[0] else c2[0] <= p[0] <= c1[0] and c2[1] <= p[1] <= c1[1]
    if len(x) != 0:
        for (x_,y_) in zip(x, y):
            for (x__,y__) in zip(x_, y_):
                if inside([x__, y__], [1690, 40], [1850, 60]):
                    cv2.rectangle(img, (1690, 40), (1850, 60), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, "River Flows in U", (1400, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
                    pressed_keys,pressed_notes = sound_to_play(River,white,black,white_piano_notes,black_piano_notes)
                if inside([x__, y__], [1690, 70], [1850, 90]):
                    cv2.rectangle(img, (1690, 70), (1850, 90), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, "Fur Elise", (1400, 90), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
                    pressed_keys,pressed_notes = sound_to_play(Fur,white,black,white_piano_notes,black_piano_notes)
                if inside([x__, y__], [1690, 100], [1850, 120]):
                    cv2.rectangle(img, (1690, 100), (1850, 120), (255, 255, 255), cv2.FILLED)
                    cv2.putText(img, "Clair de Lune", (1400, 120), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
                    pressed_keys,pressed_notes = sound_to_play(Clair,white,black,white_piano_notes,black_piano_notes)
                else:
                    continue
    return img, pressed_keys,pressed_notes

                        
                    