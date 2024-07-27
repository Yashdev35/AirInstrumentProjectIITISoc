import numpy as np
import cv2

from hand_detection import Hands_detection


def circle_fingertips(img,x,y):
        radius=4
        color=(0,0,255)
        thickness=-1
        fingertips=[4,8,12]
        if len(x)>0:
            for x_,y_ in zip(x,y):
                for tip in fingertips:
                    cv2.circle(img,(x_[tip],y_[tip]),radius,color,thickness)
        return img

def get_coordinates(shape,distance_threshold):
    hd=Hands_detection()
    cap1=cv2.VideoCapture(0)
    while True:
        _,frame=cap1.read()
        frame = cv2.resize(frame, (shape[0],shape[1]))
        frame = cv2.flip(frame, 1)
        hd.detect(frame)
        frame=circle_fingertips(frame,hd.x,hd.y)
        cv2.imshow('Piano Configuration',frame)
        cv2.waitKey(1)
        if len(hd.x) != 0:
            rect_width = 1600
            rect_height = 350
            [center_x, center_y] = hd.x[0][8], hd.y[0][8]
            [forefinger_x0,forefinger_y0] = hd.x[0][8],hd.y[0][8]
            [middlefinger_x0,middlefinger_y0] = hd.x[0][12],hd.y[0][12]
            cv2.rectangle(frame,pt1 = (int((center_x) - (rect_width / 2)), int((center_y) + (rect_height / 2))),
                          pt2=(int((center_x) + (rect_width / 2)), int((center_y) - (rect_height / 2)))
                          , color=(0,0,0), thickness=1)
            cv2.imshow('Piano Configuration',frame)
            cv2.waitKey(1)
            distance = cv2.norm(np.array((forefinger_x0,forefinger_y0)), np.array((middlefinger_x0,middlefinger_y0)), cv2.NORM_L2)
            if distance < distance_threshold:
                pts = [
                [[((center_x) - (rect_width / 2)), ((center_y) - (rect_height / 2))]],
                [[((center_x) + (rect_width / 2)), ((center_y) - (rect_height / 2))]],
                [[((center_x) + (rect_width / 2)), ((center_y) + (rect_height / 2))]],
                [[((center_x) - (rect_width / 2)), ((center_y) + (rect_height / 2))]]
                ]
                cap1.release()
                cv2.destroyAllWindows()
                return pts

def piano_configuration(shape,distance_threshold):
    pts=get_coordinates(shape,distance_threshold)
    return pts