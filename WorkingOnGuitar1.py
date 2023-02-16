# This Module is for the Air Guitar Project
# Current Working Version: 1.0
# Features:
# - Detects the hand and draws the landmarks
# - Draws the lines between the landmarks
# - Calculates the distance between the landmarks
# - Draws a line from the wrist to the index finger
# - Offsets line if required

# TODO:
# select hands
# - IMPORTANT: Make it all Dynamic based on either z value or distance between wrist and index finger (some sort of triangulation)
#   - Multiply the POINT dictionary by image.shape() {[0],[1]} and remove the confusing code from the DrawLine function
# - Set up a intial picture to initalize the point dictionary
# - Fix the point dictionary to be more dynamic(multiply with image.shape() {[0],[1]}) and easier to use 
# - Remove the hand landmarks from the image
# - Overall Clean up the code

# MAYBE:
# - Add a function to calculate the angle between the lines
# - Add a function to calculate the distance between the lines

# DONE:


import cv2
import mediapipe as mp
import pyautogui as gui
import math

def rainbow_gradient(num_colors):
    """Returns a list of RGB colors that make a rainbow gradient"""

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
def DistanceBetweenTwoPoints(point1, point2):
    dist = ((point1.x - point2.x)**2 + (point1.y - point2.y)**2)**0.5
    # # take z vaule into account
    # dist = dist + (point1.z - point2.z)**2
    
    # normalize using z value as reference
    # dist = dist / -point1.z


    return dist
def ResizeWithAspectRatio(image, width = 1920, height= 1080, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    r = width / float(w)
    dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)
def initializeCode(hands):
    success, image = cap.read()
    image =  cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # TODO: Make it Raise an error Later

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(image)

    return image, results
def endCode():
    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        return True
    else:
        return False
def markHands(image, results, point, mp_drawing, mp_drawing_styles, mp_hands):
    # making the marks on the hands
    for hand_no,hand_landmarks in enumerate(results.multi_hand_landmarks):
        for part,vals in point.items():
            h , w , c = image.shape
            cx , cy = int(vals.x * w) , int(vals.y * h)
            cv2.putText(image,"{:.2f} {:.2f}".format(vals.x,vals.y), (cx, cy), cv2.FONT_HERSHEY_PLAIN, vals.z * -5, (0, 0, 255), 1)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
def showImage(image):
    # resize image
    image = ResizeWithAspectRatio(image, width=int(gui.size()[0]/2), height=int(gui.size()[1]/2))
    cv2.imshow('MediaPipe Hands', image)
    return
def DrawLine(image,point1,point2,length = 3 ,xoffset = 0, yoffset = 0 ,Lncolor = (255,0,0) , Hand = True , OneHand = False):
    # TODO: Maybe make the lenght percentage based on the distance between the points or completely fix the line length
    if OneHand:
        startpoint = (int(point1.x * 640) + xoffset ,int(point1.y * 480) + yoffset)
        endpoint = point2
    elif Hand:
        startpoint = (int(point1.x * 640) + xoffset ,int(point1.y * 480) + yoffset)
        endpoint = (int(point2.x * 640) + xoffset ,int(point2.y * 480) + yoffset)

        endpoint = (int(endpoint[0] + (endpoint[0] - startpoint[0]) * length),int(endpoint[1] + (endpoint[1] - startpoint[1]) * length))
        
    else:
        startpoint = point1
        endpoint = point2
        
    cv2.line(image, startpoint, endpoint, Lncolor, 5)
    return
    # return startpoint, endpoint
def DrawBoard(image, point , lines = 4, xoffset = 10, yoffset = 10):
    # Draw the board
    col = rainbow_gradient(lines)
    for stringLine in range(lines):
        DrawLine(image, point["Wrist"], point["Middle MCP"], length=4, xoffset=stringLine * xoffset, yoffset= stringLine * yoffset, Lncolor=col[stringLine])

        
  
def main():

    global cap

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands


    # For webcam input:
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    while cap.isOpened():
                    
        image, results = initializeCode(hands)

        if results.multi_hand_landmarks:
            point = {"Wrist" : results.multi_hand_landmarks[0].landmark[0] ,"Thumb CMC" : results.multi_hand_landmarks[0].landmark[1], \
                "Thumb MCP" : results.multi_hand_landmarks[0].landmark[2], "Thumb IP" : results.multi_hand_landmarks[0].landmark[3], \
                    "Thumb Tip" : results.multi_hand_landmarks[0].landmark[4], "Index MCP" : results.multi_hand_landmarks[0].landmark[5], \
                        "Index PIP" : results.multi_hand_landmarks[0].landmark[6], "Index DIP" : results.multi_hand_landmarks[0].landmark[7], \
                            "Index Tip" : results.multi_hand_landmarks[0].landmark[8], "Middle MCP" : results.multi_hand_landmarks[0].landmark[9], \
                                "Middle PIP" : results.multi_hand_landmarks[0].landmark[10], "Middle DIP" : results.multi_hand_landmarks[0].landmark[11], \
                                    "Middle Tip" : results.multi_hand_landmarks[0].landmark[12], "Ring MCP" : results.multi_hand_landmarks[0].landmark[13], \
                                        "Ring PIP" : results.multi_hand_landmarks[0].landmark[14], "Ring DIP" : results.multi_hand_landmarks[0].landmark[15], \
                                            "Ring Tip" : results.multi_hand_landmarks[0].landmark[16], "Pinky MCP" : results.multi_hand_landmarks[0].landmark[17], \
                                                "Pinky PIP" : results.multi_hand_landmarks[0].landmark[18], "Pinky DIP" : results.multi_hand_landmarks[0].landmark[19], \
                                                    "Pinky Tip" : results.multi_hand_landmarks[0].landmark[20]}
            markHands(image, results, point,mp_drawing,mp_drawing_styles,mp_hands)
    
            # DrawBoard(image, point ,lines=1)
            DrawLine(image, point["Wrist"], point["Middle MCP"], length=4, xoffset=0, yoffset= 0, Lncolor=(255,0,0), Hand = True, OneHand = True)
            # s,e = DrawLine(image, point["Wrist"], point["Middle MCP"], length=4, xoffset=0, yoffset= 0, Lncolor=(255,0,0), Hand = True, OneHand = True)
            # m = (e[1] - s[1]) / (e[0] - s[0])
            # b = s[1] - m * s[0]
            # if b == 0:
            #     print("Horizontal")

        showImage(image)
        endCode()


if __name__ == "__main__":
    main()

    