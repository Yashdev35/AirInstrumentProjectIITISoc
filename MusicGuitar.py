import cv2
import mediapipe as mp
from pyautogui import size 
from math import sin, cos, atan2, pi
from pygame import mixer
import numpy as np
from volumeguesturecontrol_guitar import volControl

class VirGuitar():
        def __init__(self):
            self.SoundChannel, self.sound = self.initializeMusic()
            self.color = (0, 0, 0)
            self.ExplicitMarking = True
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_hands = mp.solutions.hands
            self.cap = cv2.VideoCapture(0)
            self.hands = self.mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
            if self.ExplicitMarking:
                self.OneHandExist = False

            self.start() 

        def rainbow_gradient(self,num_colors):
            colors = []
            for i in range(num_colors):
                r, g, b = 0, 0, 0
                if 0 <= i < num_colors/6:
                    r = 255
                    g = int(255*i/(num_colors/6))
                elif num_colors/6 <= i < num_colors/3:
                    r = 255 - int(255*(i-num_colors/6)/(num_colors/6))
                    g = 255
                elif num_colors/3 <= i < num_colors/2:
                    g = 255
                    b = int(255*(i-num_colors/3)/(num_colors/6))
                elif num_colors/2 <= i < 2*num_colors/3:
                    g = 255 - int(255*(i-num_colors/2)/(num_colors/6))
                    b = 255
                elif 2*num_colors/3 <= i < 5*num_colors/6:
                    r = int(255*(i-2*num_colors/3)/(num_colors/6))
                    b = 255
                elif 5*num_colors/6 <= i < num_colors:
                    r = 255
                    b = 255 - int(255*(i-5*num_colors/6)/(num_colors/6))
                colors.append((r, g, b))
            return colors



        def ResizeWithAspectRatio(self,image, width = 1920, height= 1080, inter=cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape[:2]
            r = width / float(w)
            dim = (width, int(h * r))
            return cv2.resize(image, dim, interpolation=inter)

        def MusicFileNameBuilder(self,string,folder = "Media"):
            stringToString = { 0 : "E2", 1 : "A2", 2 : "D3", 3 : "G3", 4 : "B3", 5 : "E4"}
            fileName = folder + "/" + stringToString[string] + ".wav"
            return fileName

        def initializeMusic(self,folder = 'samples_guitar'):
            mixer.init()
            stringToString = { 0 : "E2", 1 : "A2", 2 : "D3", 3 : "G3", 4 : "B3", 5 : "E4"}
            SoundChannel = {}
            Sound = {}
            for key, value in stringToString.items():
                SoundChannel[key] = mixer.Channel(key)
                Sound[key] = mixer.Sound(folder+"/" + value.strip() + ".wav")
            return SoundChannel,Sound

        def initializeHandsCode(self, hands,cap):
            success, image = cap.read()
            image =  cv2.flip(image, 1)
            if not success:
                print("Ignoring empty camera frame.")
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = hands.process(image)
            return image, results

        def endCode(self,cap,hands,debug = False):
            if debug:
                WaitVal = 0
            else:
                WaitVal = 1
            if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(WaitVal) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                hands.close()
                return True

        def markHands(self,image, results, point, mp_drawing, mp_drawing_styles, mp_hands):

            for hand_no,hand_landmarks in enumerate(results.multi_hand_landmarks):
                for part,vals in point.items():
                    h , w , c = image.shape
                    cx , cy = int(vals.x * w) , int(vals.y * h)
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        def showImage(self,image):
            image = self.ResizeWithAspectRatio(image, width=int(size()[0]/2), height=int(size()[1]/2))
            cv2.imshow('MediaPipe Hands', image)
            return

        def DrawLine(self,image,point1,point2,length = 3 ,xoffset = 0, yoffset = 0,offset = 0 ,Lncolor = (255,0,0) , Hand = True , OneHand = False, log = False,thick=5 ):
            if offset != 0:
                xoffset = offset
                yoffset = offset
            if OneHand:
                startpoint = (int(point1.x * image.shape[1]) + xoffset ,int(point1.y * image.shape[0]) + yoffset)
                endpoint = point2
            elif Hand:
                startpoint = (int(point1.x * image.shape[1]) + xoffset ,int(point1.y * image.shape[0]) + yoffset )
                endpoint = (int(point2.x * image.shape[1]) + xoffset ,int(point2.y * image.shape[0]) + yoffset )
                endpoint = (int(endpoint[0] + (endpoint[0] - startpoint[0]) * length),int(endpoint[1] + (endpoint[1] - startpoint[1]) * length))
            else:
                startpoint = point1
                endpoint = point2
            if log:
                print("startpoint: {}, endpoint: {}".format(startpoint,endpoint))
            cv2.line(image, startpoint, endpoint, Lncolor, thickness=thick)
            return startpoint, endpoint

        def NewPos(self,image,AnchorStartPoint,AnchorEndPoint,offset = 20,length = 4 , absoulute_offset = 40, absolute_length = False):
            x1,y1 = int(AnchorStartPoint.x * image.shape[1]), int(AnchorStartPoint.y * image.shape[0])
            x2,y2 = int(AnchorEndPoint.x * image.shape[1]), int(AnchorEndPoint.y * image.shape[0])

            Angle = atan2(y2 - y1, x2 - x1)

            x3 = x1 - offset * cos(pi/2 - Angle)
            y3 = y1 + offset * sin(pi/2 - Angle)

            x3 = int(x3)
            y3 = int(y3)
            
            startpoint = (x3,y3)
            
            x4 = x2 - offset * cos(pi/2 - Angle)
            y4 = y2 + offset * sin(pi/2 - Angle)
            
            x4 = int(x4)
            y4 = int(y4)
            
            endpoint = (x4,y4)
            if absolute_length:
                endpoint = ( int(endpoint[0] + absoulute_offset * cos(Angle)) , int( endpoint[1] + absoulute_offset * sin(Angle) ) ) 
            else:
                endpoint = (int(endpoint[0] + (endpoint[0] - startpoint[0]) * length),int(endpoint[1] + (endpoint[1] - startpoint[1]) * length))
            return startpoint, endpoint

        def LineOffset(self,x1,y1,x2,y2,value = 0):

            if value == 0:
                return (x2,y2)

            Angle = atan2(y2 - y1, x2 - x1)

            x2 = int(x2 + value * cos(Angle))
            y2 = int(y2 + value * sin(Angle))

            return (x2,y2)

        def DrawBoard(self,image, AnchorStartPoint,AnchorEndPoint , lines = 4, offset = 0,dynamic = False,log = False , length = 2, thickness = 4 , absolute_length = False, absoulute_offset = 0 , fretOffset = 0):

            col = self.rainbow_gradient(lines)

            posList = []

            if dynamic:
                offset = abs(AnchorEndPoint.z ** 0.2 * 25) 
                length = abs(AnchorEndPoint.z ** 0.2 * length * 3)
                if log:
                    print("Offset: {}".format(offset)) 
                    print("Length: {}".format(length))
            Stoffset = -(lines//2) * offset

            for stringLine in range(lines):
                start,end = self.NewPos(image,AnchorStartPoint,AnchorEndPoint,offset = Stoffset, absolute_length=absolute_length , absoulute_offset = absoulute_offset , length = length)
                start = self.LineOffset(end[0],end[1],start[0],start[1],value = fretOffset)
                self.DrawLine(image, start, end, length=length,Hand=False,log=log,Lncolor=col[stringLine],thick=thickness)
                posList.append((start,end))
                Stoffset += offset

                if log:
                    print("String {} => Start {} --> End {}".format(stringLine, start, end))

            return posList

        def ContactCheck(self,image,draw, finger,sound,SoundChannel,log = False, accuracy = 10, logSize = 0.75, logColor = (0, 255, 0)):
            thumb_x = int(finger.x * image.shape[1])
            thumb_y = int(finger.y * image.shape[0])
            thumb_z = abs(finger.z) / accuracy
            for GuitarString in range(len(draw)):
                x1 = int(draw[GuitarString][0][0])
                y1 = int(draw[GuitarString][0][1])
                x2 = int(draw[GuitarString][1][0])
                y2 = int(draw[GuitarString][1][1])
                if log:
                    print("String {} => Start {} --> End {}".format(GuitarString, (x1, y1), (x2, y2)))
                    print("String {} => Start {} --> End {}".format(GuitarString, draw[GuitarString][0], draw[GuitarString][1]))

                if abs((y2 - y1) * thumb_x - (x2 - x1) * thumb_y + x2 * y1 - y2 * x1) / ((y2 - y1) ** 2 + (x2 - x1) ** 2) <= thumb_z:
                    SoundChannel[GuitarString].play(sound[GuitarString])

        def DrawBoardLog(self,draw):
            for i in range(len(draw)):
                print("Line {} => Start {} --> End {}".format(i, draw[i][0], draw[i][1]))

        def AddBlobToFinger(self,image, finger, color = (0, 0, 255), radius = 25,log = False , thickness = 2):
            if log:
                print("Finger => x: {} y: {} z: {}".format(finger.x, finger.y, finger.z))
                print("coords: {}, {}".format(int(finger.x * image.shape[1]), int(finger.y * image.shape[0])))
                print("image shape: {}".format(image.shape))
                print("color: {}".format(color))
                print("radius: {}".format(radius))

            radius = int(radius * abs(finger.z) ** 0.2)
            cv2.circle(image, (int(finger.x * image.shape[1]), int(finger.y * image.shape[0])), radius, color,thickness)
        
        def FingerLog(self,point,finger):
            print("Finger {} => x: {} y: {} z: {}".format(finger,point[finger].x, point[finger].y,point[finger].z))
            
        guitar_image = cv2.imread('guitar.png', cv2.IMREAD_UNCHANGED)  # Ensure the guitar image has an alpha channel

        def overlayGuitar(self, image, wrist1, wrist2, guitar_image):
            try:
                wrist_coords = (int(wrist1.x * image.shape[1]), int(wrist1.y * image.shape[0]))
                index_tip_coords = (int(wrist2.x * image.shape[1]), int(wrist2.y * image.shape[0]))

                angle = np.arctan2(index_tip_coords[1] - wrist_coords[1], index_tip_coords[0] - wrist_coords[0]) * 180 / np.pi

                angle += 0 

                guitar_image = cv2.flip(guitar_image, 1)

                distance = int(np.linalg.norm(np.array(wrist_coords) - np.array(index_tip_coords)))

                size_multiplier = 1.2  # Increase or decrease this value to change the size

                scaling_factor = (distance / guitar_image.shape[1]) * size_multiplier
                guitar_resized = cv2.resize(guitar_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                center = (guitar_resized.shape[1] // 2, guitar_resized.shape[0] // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                guitar_rotated = cv2.warpAffine(guitar_resized, rotation_matrix, (guitar_resized.shape[1], guitar_resized.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                overlay_x = wrist_coords[0] - guitar_rotated.shape[1] // 2
                overlay_y = wrist_coords[1] - guitar_rotated.shape[0] // 2

                overlay_x = max(0, min(overlay_x, image.shape[1] - guitar_rotated.shape[1]))
                overlay_y = max(0, min(overlay_y, image.shape[0] - guitar_rotated.shape[0]))

                overlay_w = min(guitar_rotated.shape[1], image.shape[1] - overlay_x)
                overlay_h = min(guitar_rotated.shape[0], image.shape[0] - overlay_y)

                alpha_mask = guitar_rotated[:overlay_h, :overlay_w, 3] / 255.0
                alpha_inv = 1.0 - alpha_mask

                for c in range(3):
                    image[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c] = \
                        (alpha_mask * guitar_rotated[:overlay_h, :overlay_w, c] +
                        alpha_inv * image[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c])

                return image
            except KeyError as e:
                print(f"KeyError: {e} - Check if landmarks contain the required keys.")
                return image

        def start(self):
            while self.cap.isOpened():
                image, results = self.initializeHandsCode(self.hands, self.cap)

                self.color = (0, 255, 0)
                initial_y_position = 30
                line_height = 25

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                    y_position = initial_y_position
                    cv2.putText(image, "Use one hand for volume control.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Put the other hand to start playing guitar.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Use thumb and index finger to control volume.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Keep them close to the camera.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    landmarks = results.multi_hand_landmarks
                    image = volControl(image, landmarks)

                if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                    y_position = initial_y_position
                    cv2.putText(image, "To play guitar, put both hands in front.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Rotate the hand with strings to connect.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Use the finger with the red dot to play.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    y_position += line_height
                    cv2.putText(image, "Remove one hand from frame for volume control.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color, 2)
                    
                    left = {"Wrist" : results.multi_hand_landmarks[0].landmark[self.mp_hands.HandLandmark.WRIST], "Thumb CMC" : results.multi_hand_landmarks[0].landmark[1], \
                            "Thumb MCP" : results.multi_hand_landmarks[0].landmark[2], "Thumb IP" : results.multi_hand_landmarks[0].landmark[3], \
                            "Thumb TIP" : results.multi_hand_landmarks[0].landmark[4], "Index MCP" : results.multi_hand_landmarks[0].landmark[5], \
                            "Index PIP" : results.multi_hand_landmarks[0].landmark[6], "Index DIP" : results.multi_hand_landmarks[0].landmark[7], \
                            "Index TIP" : results.multi_hand_landmarks[0].landmark[8], "Middle MCP" : results.multi_hand_landmarks[0].landmark[9], \
                            "Middle PIP" : results.multi_hand_landmarks[0].landmark[10], "Middle DIP" : results.multi_hand_landmarks[0].landmark[11], \
                            "Middle TIP" : results.multi_hand_landmarks[0].landmark[12], "Ring MCP" : results.multi_hand_landmarks[0].landmark[13], \
                            "Ring PIP" : results.multi_hand_landmarks[0].landmark[14], "Ring DIP" : results.multi_hand_landmarks[0].landmark[15], \
                            "Ring TIP" : results.multi_hand_landmarks[0].landmark[16], "Pinky MCP" : results.multi_hand_landmarks[0].landmark[17], \
                            "Pinky PIP" : results.multi_hand_landmarks[0].landmark[18], "Pinky DIP" : results.multi_hand_landmarks[0].landmark[19], \
                            "Pinky TIP" : results.multi_hand_landmarks[0].landmark[20]}
                    
                    right = {"Wrist": results.multi_hand_landmarks[1].landmark[self.mp_hands.HandLandmark.WRIST], "Thumb CMC" : results.multi_hand_landmarks[1].landmark[1], \
                            "Thumb MCP" : results.multi_hand_landmarks[1].landmark[2], "Thumb IP" : results.multi_hand_landmarks[1].landmark[3], \
                            "Thumb TIP" : results.multi_hand_landmarks[1].landmark[4], "Index MCP" : results.multi_hand_landmarks[1].landmark[5], \
                            "Index PIP" : results.multi_hand_landmarks[1].landmark[6], "Index DIP" : results.multi_hand_landmarks[1].landmark[7], \
                                "Index TIP" : results.multi_hand_landmarks[1].landmark[8], "Middle MCP" : results.multi_hand_landmarks[1].landmark[9], \
                                "Middle PIP" : results.multi_hand_landmarks[1].landmark[10], "Middle DIP" : results.multi_hand_landmarks[1].landmark[11], \
                                "Middle TIP" : results.multi_hand_landmarks[1].landmark[12], "Ring MCP" : results.multi_hand_landmarks[1].landmark[13], \
                                "Ring PIP" : results.multi_hand_landmarks[1].landmark[14], "Ring DIP" : results.multi_hand_landmarks[1].landmark[15], \
                                "Ring TIP" : results.multi_hand_landmarks[1].landmark[16], "Pinky MCP" : results.multi_hand_landmarks[1].landmark[17], \
                                "Pinky PIP" : results.multi_hand_landmarks[1].landmark[18], "Pinky DIP" : results.multi_hand_landmarks[1].landmark[19], \
                                "Pinky TIP" : results.multi_hand_landmarks[1].landmark[20]}
                    
                    left_right = left["Wrist"]
                    right_wrist = right["Wrist"]

                    image = self.overlayGuitar(image, left_right, right_wrist, self.guitar_image)

                    if self.ExplicitMarking:
                        self.OneHandExist = True
                        self.markHands(image, results, left, mp_drawing=self.mp_drawing, mp_drawing_styles=self.mp_drawing_styles, mp_hands=self.mp_hands)

                    draw = self.DrawBoard(image, right["Index MCP"], right["Pinky MCP"], lines=6, length=4, offset=20, log=False, dynamic=True, fretOffset=20)
                    self.AddBlobToFinger(image, left["Index TIP"])
                    self.ContactCheck(image, draw, left["Index TIP"], sound=self.sound, SoundChannel=self.SoundChannel, log=False, accuracy=5, logColor=self.color)

                elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                    if self.ExplicitMarking and self.OneHandExist:
                        self.markHands(image, results, left, mp_drawing=self.mp_drawing, mp_drawing_styles=self.mp_drawing_styles, mp_hands=self.mp_hands)

                    cv2.putText(image, "One Hand Detected", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.color, 2)

                else:
                    cv2.putText(image, "No Hands Detected", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, self.color, 2)

                cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                self.showImage(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.endCode(self.cap, self.hands, debug=False)    
    # ------------------------------------------------------------------------------------------------


def main():
    VirGuitar()

if __name__ == "__main__":
    main()    