import numpy as np
import cv2
import time


from keyboard import Piano
from piano_config import piano_configuration
from hand_detection import Hands_detection
from check_key import check_keys,tap_detection
from piano_sound import play_piano_sound
from volumegesturecontrol import volControl
from autoplay import auto_play


class VirPiano():
    def __init__(self,num_octaves=2,list_of_octaves=[0,8],height_and_width_black=[[5,8],[5,8]],shape=(1880,680,3),tap_threshold=20,piano_config_threshold=30,piano_config=1):
        self.hand_detection=Hands_detection()
        self.shape=shape
        self.image=np.zeros(self.shape, np.uint8)
        self.num_octaves=num_octaves
        self.list_of_octaves=list_of_octaves
        self.height_and_width_black=height_and_width_black
        self.tap_threshold=tap_threshold
        self.piano_config_threshold=piano_config_threshold
        self.pts=np.array([[[100,350]],[[700,350]],[[700,550]],[[100,550]]])
        self.piano_keyboard=Piano(self.pts)
        self.piano_config=piano_config
        self.x=[]
        self.y=[]
        self.previous_x=[]
        self.previous_y=[]
        self.white_piano_notes,self.black_piano_notes=self.get_piano_notes()
        self.start()

    def get_piano_notes(self):
        white_piano_notes = [
            'A0', 'B0',
            'C1', 'D1', 'E1', 'F1', 'G1', 'A1', 'B1',
            'C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2',
            'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
            'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
            'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5',
            'C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6',
            'C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7',
            'C8'
        ]

        black_piano_notes = [
        'Bb0',
        'Db1', 'Eb1', 'Gb1', 'Ab1', 'Bb1',
        'Db2', 'Eb2', 'Gb2', 'Ab2', 'Bb2',
        'Db3', 'Eb3', 'Gb3', 'Ab3', 'Bb3',
        'Db4', 'Eb4', 'Gb4', 'Ab4', 'Bb4',
        'Db5', 'Eb5', 'Gb5', 'Ab5', 'Bb5',
        'Db6', 'Eb6', 'Gb6', 'Ab6', 'Bb6',
        'Db7', 'Eb7', 'Gb7', 'Ab7', 'Bb7'
        
        ]


        return white_piano_notes,black_piano_notes

    def circle_fingertips(self,img):
        radius=4
        color=(0,0,255)
        thickness=-1
        fingertips=[4,8,12,16,20]
        if len(self.x)>0:
            for x,y in zip(self.x,self.y):
                for tip in fingertips:
                    cv2.circle(img,(x[tip],y[tip]),radius,color,thickness)
        return img

    def start(self):
        cap=cv2.VideoCapture(0)
        previousTime=0
        while True:
            _,frame=cap.read()
            frame = cv2.resize(frame, (self.shape[0],self.shape[1]))
            frame = cv2.flip(frame, 1)

            while self.piano_config==1:
                self.pts=piano_configuration(self.shape,self.piano_config_threshold)
                self.piano_keyboard=Piano(self.pts)
                self.piano_config=0
                cap=cv2.VideoCapture(0)

            self.image=frame.copy()
            self.hand_detection.detect(frame)
            self.x=self.hand_detection.x
            self.y=self.hand_detection.y

            self.image=self.piano_keyboard.make_keyboard(self.image)
            self.image=self.piano_keyboard.print_notes(self.image)

            pressed_keys={"White":[],"Black":[]}
            pressed_notes=[]
            pressed_keys1={"White":[],"Black":[]}
            pressed_notes1=[]
            if len(self.previous_x)==len(self.x):
                for (x_,y_),(previous_x_,previous_y_) in zip(zip(self.x,self.y),zip(self.previous_x,self.previous_y)):
                    tapped_keys=tap_detection(previous_y_,y_,self.tap_threshold)
                    keys,notes=check_keys(x_,y_,self.piano_keyboard.white,self.piano_keyboard.black,self.white_piano_notes,self.black_piano_notes,tapped_keys)
                    for note in notes:
                        pressed_notes.append(note)
                    for w in keys['White']:
                        pressed_keys['White'].append(w)
                    for b in keys['Black']:
                        pressed_keys['Black'].append(b)

            self.image=self.piano_keyboard.change_color(self.image,pressed_keys)


            self.image=self.circle_fingertips(self.image)


            self.previous_x=self.x
            self.previous_y=self.y

            if len(pressed_notes)>0:
                play_piano_sound(pressed_notes)

            currentTime = time.time()
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime
            
            cv2.putText(self.image, str(int(fps)) + "FPS", (10, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
            cv2.putText(self.image, '+', (700, 55), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 0, 0), 3)
            cv2.putText(self.image, '-', (700, 95), cv2.FONT_HERSHEY_TRIPLEX, 1, (128, 0, 0), 3)
            cv2.circle(self.image, (710, 60), 40, (0,0,0), thickness=1)
            
            self.image = volControl(self.image, self.x, self.y)
            
            cv2.putText(self.image, "River Flows in You", (1400, 50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
            cv2.putText(self.image, "Für Elise", (1400, 80), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
            cv2.putText(self.image, "Clair de Lune”", (1400, 110), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1)
            cv2.rectangle(self.image, (1690, 40), (1850, 60), (0, 0, 0), 3)
            cv2.rectangle(self.image, (1690, 70), (1850, 90), (0, 0, 0), 3)
            cv2.rectangle(self.image, (1690, 100), (1850, 120), (0, 0, 0), 3)
            
            pressed_keys2 = {"White":[],"Black":[]}
            self.image, pressed_keys1, pressed_notes1 = auto_play(self.x, self.y, self.image,self.piano_keyboard.white,self.piano_keyboard.black,self.white_piano_notes,self.black_piano_notes)    
            if len(pressed_notes1)>0:
                for i in range(len(pressed_notes1)):
                    pressed_keys2['White'].append(pressed_keys1["White"][i])
                    #pressed_keys2['Black'].append(pressed_keys1["Black"][i])
                    time.sleep(0.2)
                    self.image=self.piano_keyboard.change_color(self.image,pressed_keys2)
                    play_piano_sound(pressed_notes1[i])
                    pressed_keys2 = {"White":[],"Black":[]}
            
            cv2.namedWindow("Hand detection", cv2.WINDOW_NORMAL)
            cv2.imshow('Hand detection', self.image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
def main():
    VirPiano()

if __name__ == "__main__":
    main()
