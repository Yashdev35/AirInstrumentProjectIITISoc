import cv2
import numpy as np

class piano():
    def __init__(self,pts,num_octaves,num_white_keys = 52, num_black_keys = 36):
        self.pts=pts
        self.num_octaves=num_octaves
        self.num_white_keys = num_white_keys
        self.num_black_keys = num_black_keys
        self.white=[]
        self.black=[]
        self.get_keyboard_keys()
    def get_keyboard_keys(self):

        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]= self.pts
        piano_width = x2- x0
        piano_height = y2 - y0
        
        white_key_width = piano_width // self.num_white_keys
        black_key_width = 5*white_key_width // 8
        white_key_height = piano_height
        black_key_height = 5*piano_height // 8
        
        for i in range(self.num_white_keys):
            x_start = x1 + i * white_key_width
            y_start = y1
            x_end = x_start + white_key_width
            y_end = y1 + white_key_height
            
            
            self.white.append([[x_start, y_start], [x_end, y_end], [x_start, y_end], [x_end, y_start]])
            
        for i in range(self.num_black_keys):  # One less black key than white keys
            if i % 7 != 2 and i % 7 != 6:
                x_start = x1 + (i + 1) * white_key_width - black_key_width // 2
                y_start = y1
                x_end = x_start + black_key_width
                y_end = y1 + black_key_height
                
                self.black.append([[x_start, y_start], [x_end, y_end], [x_start, y_end], [x_end, y_start]])
    
    def print_notes(self,img):
        [[[x0,y0]],[[x1,y1]],[[x2,y2]],[[x3,y3]]]= self.pts
        piano_width = x2- x0
        
        white_key_width = piano_width // self.num_white_keys
        
        note_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        octave = 0
        for i in range(self.num_white_keys):
            x_start = x1 + i * white_key_width
            y_start = y1
            cv2.putText(img, f'{note_names[i % 7]}{octave}', (x_start + 5, y_start + 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            if note_names[i % 7] == 'G':
                octave += 1
        return img
    
    def make_keyboard(self,img):
        img=cv2.fillPoly(img,self.white,(255,255,255))
        img=cv2.polylines(img,self.white,True,(0,0,0),2)
        img=cv2.fillPoly(img,self.black,(0,0,0))
        return img
    
    def change_color(self,img,pressed_keys):
        white_keys=[self.white[i] for i in pressed_keys['White']]
        black_keys=[self.black[i] for i in pressed_keys['Black']]
        img=cv2.fillPoly(img,white_keys,(0,255,0))
        img=cv2.polylines(img,self.white,True,(0,0,0),2)
        img=cv2.fillPoly(img,self.black,(0,0,0))
        img=cv2.fillPoly(img,black_keys,(128,128,128))
        return img

if __name__ == "__main__":

    frame = np.zeros((680, 1880, 3), dtype=np.uint8)  # Example black frame
    

    

    cv2.imshow("Piano Keys", frame_with_keys)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
