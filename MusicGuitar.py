# This Module is for the Air Guitar Project
# Current Working Version: 3.5
# Features:
# - Uses Index Finger for Strumming
# - Plays the String Sound when the Index Finger is near the String
# - Use Channels for better Sound Quality
# - Use better Sound Files
# - Fixed the dynamic string offset to be more smooth
# TODO:
# Setup Chord Detection Mechanism
# Setup Chords
# Update Documentation

# Changelog:
# - Removed the Distance Function

import cv2
import mediapipe as mp
from pyautogui import size 
from math import sin, cos, atan2, pi
from pygame import mixer
import numpy as np
from volumegesturecontrol import volControl
def rainbow_gradient(num_colors):
    """
    -------------------------------------------------------------
    ### Returns a list of RGB colors that make a rainbow gradient
    -------------------------------------------------------------
    ### Parameters:
        num_colors: Number of colors in the gradient [int]

    ### Returns:
        colors: List of RGB colors [(int, int, int)]
    """
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



def ResizeWithAspectRatio(image, width = 1920, height= 1080, inter=cv2.INTER_AREA):
    '''
    -------------------------------------------------------------
    Resizes the image to the given width and height
    -------------------------------------------------------------
    ### Parameters:
        image: Image to be resized [numpy array] 
        width: Width of the image [int] (Default = 1920)
        height: Height of the image [int] (Default = 1080)
        inter: Interpolation method [int] (Default = cv2.INTER_AREA)

    ### Returns:
        dim: Resized image [numpy array]
    '''
    dim = None
    (h, w) = image.shape[:2]

    r = width / float(w)
    dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def MusicFileNameBuilder(string,folder = "Media"):
    '''
    WARNING: CURRENTLY NOT IN USE
    -------------------------------------------------------------
    Builds the file name for the string sound
    -------------------------------------------------------------
    ### Parameters:
        string: String number [int]
        folder: Folder name [str] (Default = "Media")

    ### Returns:
        fileName: File name [str]
    '''
    stringToString = { 0 : "E2", 1 : "A2", 2 : "D3", 3 : "G3", 4 : "B3", 5 : "E4"}
    fileName = folder + "/" + stringToString[string] + ".wav"
    return fileName

def initializeMusic(folder = 'samples'):
    '''
    -------------------------------------------------------------
    Initializes the music 
    -------------------------------------------------------------
    ### Parameters:
        None

    ### Returns:    
        SoundChannel: Sound channel [Dictionary{int: pygame.mixer.Channel}]
        sound: Sound object [pygame.mixer.Sound]

    '''
    mixer.init()
    stringToString = { 0 : "E2", 1 : "A2", 2 : "D3", 3 : "G3", 4 : "B3", 5 : "E4"}
    SoundChannel = {}
    Sound = {}
    for key, value in stringToString.items():
        SoundChannel[key] = mixer.Channel(key)
        # fix this
        Sound[key] = mixer.Sound(folder+"/" + value.strip() + ".wav")
    return SoundChannel,Sound

def initializeHandsCode(hands,cap):
    '''
    -------------------------------------------------------------
    Initializes the code
        - Takes the camera input
        - Flips the image
        - Processes the image (for results)
    -------------------------------------------------------------
    ### Parameters:
        hands: Hand object [mediapipe object]
        cap: Camera object [cv2 object]

    ### Returns:
        image: Image from the camera [numpy array]
        results: Results from the image [mediapipe object] {For Hand Detection}
    '''
    success, image = cap.read()
    image =  cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        # TODO: Make it Raise an error Later
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    results = hands.process(image)
    return image, results

def endCode(cap,hands,debug = False):
    '''
    -------------------------------------------------------------
    Ends the code
        - Releases the camera
        - Destroys all the windows
    -------------------------------------------------------------
    ### Parameters:
        cap: Camera object [cv2 object]
        debug: Debug mode [bool] (Default = False) {If True, It will Wait for a Key Press to take a new Frame}

    ### Returns:
        True: If the code is ended [bool]
    '''
    if debug:
        WaitVal = 0
    else:
        WaitVal = 1

    if cv2.waitKey(1) & 0xFF == 27 or cv2.waitKey(WaitVal) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

        return True

def markHands(image, results, point, mp_drawing, mp_drawing_styles, mp_hands):
    '''
    -------------------------------------------------------------
    Marks the hands on the Image and Prints the Coordinates
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        results: Results from the image [mediapipe object] {For Hand Detection}
        point: Points on the hand [dict]
        mp_drawing: Drawing object [mediapipe object]
        mp_drawing_styles: Drawing Styles object [mediapipe object]
        mp_hands: Hands object [mediapipe object]

    ### Returns:
        None 
    '''

    # making the marks on the hands
    for hand_no,hand_landmarks in enumerate(results.multi_hand_landmarks):
        for part,vals in point.items():
            h , w , c = image.shape
            cx , cy = int(vals.x * w) , int(vals.y * h)
            # cv2.putText(image,"{:.2f} {:.2f}".format(vals.x,vals.y), (cx, cy), cv2.FONT_HERSHEY_PLAIN, vals.z * -5, (0, 0, 255), 1)

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

def showImage(image):
    '''
    -------------------------------------------------------------
    Shows the Resized Image 
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]  
        
    ### Returns:
        None
    '''
    image = ResizeWithAspectRatio(image, width=int(size()[0]/2), height=int(size()[1]/2))
    cv2.imshow('MediaPipe Hands', image)
    return

def DrawLine(image,point1,point2,length = 3 ,xoffset = 0, yoffset = 0,offset = 0 ,Lncolor = (255,0,0) , Hand = True , OneHand = False, log = False,thick=5 ):
    '''
    -------------------------------------------------------------
    Draws a line between two points
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        point1: First point [tuple(int x, int y)]
        point2: Second point [tuple(int x, int y)]
        length: Length of the line [int] (Default = 3)
        xoffset: X offset of the line [int] (Default = 0)
        yoffset: Y offset of the line [int] (Default = 0)
        offset: Offset of the line, Makes xoffset and yoffset the same value if not 0 [int] (Default = 0)
        Lncolor: Color of the line [tuple(int B, int G, int R)] (Default = (255,0,0))
        Hand: If the line is between two hands [bool] (Default = True) 
        OneHand: If the line is between one hand and a point [bool] (Default = False)
        log: If the function should print the startpoint and endpoint [bool] (Default = False)
        thick: Thickness of the line [int] (Default = 5)

    ### Returns:
        None

    '''
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

def NewPos(image,AnchorStartPoint,AnchorEndPoint,offset = 20,length = 4 , absoulute_offset = 40, absolute_length = False):
    '''
    -------------------------------------------------------------
    Calculates the new position of the line
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        AnchorStartPoint: Anchor Start Point [tuple(int x, int y)]
        AnchorEndPoint: Anchor End Point [tuple(int x, int y)]
        offset: Offset of the line [int] (Default = 20)
        length: Length of the line [int] (Default = 4)
        absoulute_offset: Absolute Offset of the line [int] (Default = 40)
        absolute_length: Absolute Length of the line [bool] (Default = False)

    ### Returns:
        startpoint: New Start Point [tuple(int x, int y)]
        endpoint: New End Point [tuple(int x, int y)]

    '''
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
        # make the endpoint increase in length by lenght being the number of pixels
        endpoint = ( int(endpoint[0] + absoulute_offset * cos(Angle)) , int( endpoint[1] + absoulute_offset * sin(Angle) ) ) 
    else:
        endpoint = (int(endpoint[0] + (endpoint[0] - startpoint[0]) * length),int(endpoint[1] + (endpoint[1] - startpoint[1]) * length))
    return startpoint, endpoint

def LineOffset(x1,y1,x2,y2,value = 0):
    '''
    -------------------------------------------------------------
    Calculates the offset of the line
    -------------------------------------------------------------
    ### Parameters:
        x1: X of the first point [int]
        y1: Y of the first point [int]
        x2: X of the second point [int]
        y2: Y of the second point [int]
        value: Value of the offset [int] (Default = 0)

    ### Returns:
        (x2,y2): Offset of the line [tuple(int x, int y)
    '''

    if value == 0:
        return (x2,y2)

    Angle = atan2(y2 - y1, x2 - x1)

    x2 = int(x2 + value * cos(Angle))
    y2 = int(y2 + value * sin(Angle))

    return (x2,y2)

def DrawBoard(image, AnchorStartPoint,AnchorEndPoint , lines = 4, offset = 0,dynamic = False,log = False , length = 2, thickness = 4 , absolute_length = False, absoulute_offset = 0 , fretOffset = 0):
    '''
    -------------------------------------------------------------
    Draws the board
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        AnchorStartPoint: Anchor Start Point [tuple(int x, int y)]
        AnchorEndPoint: Anchor End Point [tuple(int x, int y)]
        lines: Number of lines [int] (Default = 4)
        offset: Distance of offset between line [int] (Default = 0)
        dynamic: If the offset should be dynamic [bool] (Default = False)
        log: If the function should print the startpoint and endpoint [bool] (Default = False)
        length: Length of the line [int] (Default = 2)
        thickness: Thickness of the line [int] (Default = 4)
        absolute_length: If the length should be absolute [bool] (Default = False)
        absoulute_offset: Absolute Offset of the line [int] (Default = 40)

    ### Returns:
        posList: List of all the start and end points [list(tuple(tuple(int x, int y),tuple(int x, int y)))]


    '''
    # Draw the board
    col = rainbow_gradient(lines)

    posList = []

    if dynamic:
        offset = abs(AnchorEndPoint.z ** 0.2 * 25) 
        length = abs(AnchorEndPoint.z ** 0.2 * length * 3)
        if log:
            print("Offset: {}".format(offset)) 
            print("Length: {}".format(length))
    Stoffset = -(lines//2) * offset

    for stringLine in range(lines):
        start,end = NewPos(image,AnchorStartPoint,AnchorEndPoint,offset = Stoffset, absolute_length=absolute_length , absoulute_offset = absoulute_offset , length = length)
        start = LineOffset(end[0],end[1],start[0],start[1],value = fretOffset)
        DrawLine(image, start, end, length=length,Hand=False,log=log,Lncolor=col[stringLine],thick=thickness)
        posList.append((start,end))
        Stoffset += offset

        if log:
            print("String {} => Start {} --> End {}".format(stringLine, start, end))

    return posList

def ContactCheck(image,draw, finger,sound,SoundChannel,log = False, accuracy = 10, logSize = 0.75, logColor = (0, 255, 0)):
    '''
    -------------------------------------------------------------
    Checks if the finger is in contact with the board
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        draw: List of all the start and end points [list(tuple(tuple(int x, int y),tuple(int x, int y)))]
        finger: Finger to check [HandLandmark]
        sound: Sound to play [dict([pygame.mixer.Sound]])]
        log: If the function should print the startpoint and endpoint [bool] (Default = False)
        accuracy: Accuracy of the check [int] (Default = 10)
        logSize: Size of the log [float] (Default = 0.75)
        logColor: Color of the log [tuple(int B, int G, int R)] (Default = (0, 0, 255))

    ### Returns:
        GuitarString: The string the finger is in contact with [int]
    '''
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
            # note = sound[GuitarString]
            # note.play()
            cv2.putText(image, "String {}".format(GuitarString), (100 * (GuitarString), 100), cv2.FONT_HERSHEY_SIMPLEX, logSize, logColor, 2)
# ------------------------------------------------------------------------------------------------
# for debugging
def DrawBoardLog(draw):
    '''
    -------------------------------------------------------------
    Prints the start and end points of the board
    -------------------------------------------------------------
    ### Parameters:
        draw: List of all the start and end points [list(tuple(tuple(int x, int y),tuple(int x, int y)))]

    ### Returns:
        None
    '''
    for i in range(len(draw)):
        print("Line {} => Start {} --> End {}".format(i, draw[i][0], draw[i][1]))

def AddBlobToFinger(image, finger, color = (0, 0, 255), radius = 25,log = False , thickness = 2):
    '''
    -------------------------------------------------------------
    Draws a blob on the finger
    -------------------------------------------------------------
    ### Parameters:
        image: Image from the camera [numpy array]
        finger: Finger to check [HandLandmark]
        color: Color of the blob [tuple(int B, int G, int R)] (Default = (0, 0, 255))
        radius: Radius of the blob [int] (Default = 25)
        log: If the function should print the startpoint and endpoint [bool] (Default = False)

    ### Returns:
        None
    '''
    if log:
        print("Finger => x: {} y: {} z: {}".format(finger.x, finger.y, finger.z))
        print("coords: {}, {}".format(int(finger.x * image.shape[1]), int(finger.y * image.shape[0])))
        print("image shape: {}".format(image.shape))
        print("color: {}".format(color))
        print("radius: {}".format(radius))

    radius = int(radius * abs(finger.z) ** 0.2)
    cv2.circle(image, (int(finger.x * image.shape[1]), int(finger.y * image.shape[0])), radius, color,thickness)
 
def FingerLog(point,finger):
    '''
    -------------------------------------------------------------
    Prints the position of the finger
    -------------------------------------------------------------
    ### Parameters:
        point: Position of the finger [tuple(int x, int y, int z)]
        finger: Finger to check [HandLandmark]

    ### Returns:
        None
    '''
    print("Finger {} => x: {} y: {} z: {}".format(finger,point[finger].x, point[finger].y,point[finger].z))
    
# Load the guitar image
guitar_image = cv2.imread('guitar.png', cv2.IMREAD_UNCHANGED)  # Ensure the guitar image has an alpha channel
def overlayGuitar(image, wrist, index_tip, guitar_image):
    """
    Overlay a guitar image on the hand landmarks.
    :param image: The main image where the guitar will be overlayed.
    :param wrist: Wrist landmark from the hand.
    :param index_tip: Index finger tip landmark from the hand.
    :param guitar_image: The guitar image with alpha channel.
    :return: Image with the guitar overlay.
    """
    try:
        # Coordinates in the main image
        wrist_coords = (int(wrist.x * image.shape[1]), int(wrist.y * image.shape[0]))
        index_tip_coords = (int(index_tip.x * image.shape[1]), int(index_tip.y * image.shape[0]))

        # Calculate the size of the guitar based on the distance between wrist and index finger tip
        distance = int(np.linalg.norm(np.array(wrist_coords) - np.array(index_tip_coords)))
        scaling_factor = distance / guitar_image.shape[1]

        # Resize the guitar image
        guitar_resized = cv2.resize(guitar_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        # Calculate the position for overlay
        overlay_x = wrist_coords[0] - guitar_resized.shape[1] // 2
        overlay_y = wrist_coords[1] - guitar_resized.shape[0] // 2

        # Ensure the overlay position is within image boundaries
        overlay_x = max(0, min(overlay_x, image.shape[1] - guitar_resized.shape[1]))
        overlay_y = max(0, min(overlay_y, image.shape[0] - guitar_resized.shape[0]))

        # Extract the alpha channel from the guitar image for masking
        alpha_mask = guitar_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask

        # Overlay the guitar image using alpha blending
        for c in range(0, 3):
            image[overlay_y:overlay_y + guitar_resized.shape[0], overlay_x:overlay_x + guitar_resized.shape[1], c] = \
                (alpha_mask * guitar_resized[:, :, c] +
                 alpha_inv * image[overlay_y:overlay_y + guitar_resized.shape[0], overlay_x:overlay_x + guitar_resized.shape[1], c])

        return image
    except KeyError as e:
        print(f"KeyError: {e} - Check if landmarks contain the required keys.")
        return image
# ------------------------------------------------------------------------------------------------
def main():

    SoundChannel,sound = initializeMusic()
    
    color = (0, 0, 0)
    ExplicitMarking = True


    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    # For webcam input:
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    if ExplicitMarking:
        OneHandExist = False
    
    

    while cap.isOpened():
        image, results = initializeHandsCode(hands,cap)

        # Set up color and initial position
        color = (0, 255, 0)
        initial_y_position = 30
        line_height = 25

        # Check for single hand and display relevant instructions
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            y_position = initial_y_position
            cv2.putText(image, "Use one hand for volume control.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Put the other hand to start playing guitar.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Use thumb and index finger to control volume.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Keep them close to the camera.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # Call the volume control function
            landmarks = results.multi_hand_landmarks
            image = volControl(image, landmarks)

        # Check for two hands and display relevant instructions
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            y_position = initial_y_position
            cv2.putText(image, "To play guitar, put both hands in front.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Rotate the hand with strings to connect.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Use the finger with the red dot to play.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_position += line_height
            cv2.putText(image, "Remove one hand from frame for volume control.", (10, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            left = {"Wrist" : results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST], "Thumb CMC" : results.multi_hand_landmarks[0].landmark[1], \
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
            right = {"Wrist": results.multi_hand_landmarks[1].landmark[mp_hands.HandLandmark.WRIST], "Thumb CMC" : results.multi_hand_landmarks[1].landmark[1], \
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
                        # Extract specific landmarks
            wrist_landmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST]
            index_tip_landmark = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Call the overlayGuitar function with the specific landmarks
            image = overlayGuitar(image, wrist_landmark, index_tip_landmark, guitar_image)

            if ExplicitMarking:
                OneHandExist = True
                markHands(image, results, left, mp_drawing=mp_drawing, mp_drawing_styles=mp_drawing_styles, mp_hands=mp_hands)

            draw = DrawBoard(image, right["Index MCP"], right["Pinky MCP"], lines=6,length=4, offset=20,log=False , dynamic=True , fretOffset=20)
            AddBlobToFinger(image, left["Index TIP"])
            ContactCheck(image,draw, left["Index TIP"],sound=sound,SoundChannel=SoundChannel,log = False, accuracy = 5,logColor=color)

        elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            if ExplicitMarking and OneHandExist:
                markHands(image, results, left, mp_drawing=mp_drawing, mp_drawing_styles=mp_drawing_styles, mp_hands=mp_hands)

            cv2.putText(image, "One Hand Detected", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        else:
            cv2.putText(image, "No Hands Detected", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                 # Add text at the bottom of the image to inform the user about quitting
        cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)    
        showImage(image)
        # Check if 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    endCode(cap,hands,debug = False)


    


if __name__ == "__main__":
    main()
    
