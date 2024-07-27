import cv2
import mediapipe as mp


class Hands_detection():
   
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_hands.Hands()
        self.x=[]
        self.y=[]

    def detect(self,image,draw = False):
            self.x=[]
            self.y=[]
            results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmark in results.multi_hand_landmarks:
                    x,y = [],[]
                    for landmark in hand_landmark.landmark:
                        x.append(landmark.x),y.append(landmark.y)
                    height, width, _ = image.shape
                    x = [int(landmark*width) for landmark in x]
                    y = [int(landmark*height) for landmark in y]
                    self.x.append(x)
                    self.y.append(y)
                    if draw:
                        self.mp_draw.draw_landmarks(image, hand_landmark, self.mp_hands.HAND_CONNECTIONS)