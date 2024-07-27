import os
import cv2
import numpy as np
import pyautogui
import imutils
import pygame
import pyaudio
import wave
import threading

def Press(key):
    pyautogui.press(key)

# Initialize pygame mixer
pygame.mixer.init()

# Define the sound file paths
sound_files = {
    '7': 'drum-machine-music/ride.mp3',
    '8': 'drum-machine-music/ride-bell.mp3',
    '6': 'drum-machine-music/closedhithat.wav',
    '9': 'drum-machine-music/crash.mp3',
    '2': 'drum-machine-music/snare.mp3',
    '3': 'drum-machine-music/snarerim.wav',
    '4': 'drum-machine-music/hi-hat.mp3',
    '5': 'drum-machine-music/hat.mp3',
    'q': 'drum-machine-music/tom-hi.mp3',
    'w': 'drum-machine-music/tommid.wav',
    'e': 'drum-machine-music/lowtom.mp3',
    '1': 'drum-machine-music/kick.mp3'
}

# Load all sound files
sounds = {key: pygame.mixer.Sound(file) for key, file in sound_files.items()}

# Load the images
ride_img = cv2.imread('photos/ride.jpg')
ride_bell_img = cv2.imread('photos/ride_bell.jpeg')
hithat_close_img = cv2.imread('photos/hithat_close.png')
crash_img = cv2.imread('photos/crash.jpeg')
snare_img = cv2.imread('photos/snare.jpeg')
snarerim_img = cv2.imread('photos/snare_rim.png')
hithat_img = cv2.imread('photos/hi-hat.png')
hithatopen_img = cv2.imread('photos/hit_hat_open.jpeg')
tomhi_img = cv2.imread('photos/tom_hi.jpeg')
tommid_img = cv2.imread('photos/tom_mid.jpeg')
tomlow_img = cv2.imread('photos/tom_low.jpeg')
kick_img = cv2.imread('photos/kick.jpeg')

# Audio recording setup
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "saveaudio/recorded_beats.wav"

# Ensure the saveaudio directory exists
os.makedirs(os.path.dirname(WAVE_OUTPUT_FILENAME), exist_ok=True)

p = pyaudio.PyAudio()

frames = []

# Callback function for recording
def callback(in_data, frame_count, time_info, status):
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)

# Start the stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

cap = cv2.VideoCapture(0)

def resize_image_to_fit(img, target_size):
    target_height, target_width = target_size
    img_height, img_width = img.shape[:2]

    scale = min(target_width / img_width, target_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    img_resized = cv2.resize(img, (new_width, new_height))

    result = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = img_resized

    return result

def overlay_image(frame, img, position, size=(150, 150)):
    x1, y1, x2, y2 = position
    resized_img = resize_image_to_fit(img, size)
    frame[y1:y1+size[1], x1:x1+size[0]] = resized_img

def check_position(x, y, positions):
    for (x1, y1, x2, y2, label, key) in positions:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return (x1, y1, x2, y2, label, key)
    return None

positions = [
    (0, 600, 150, 750, 'RIDE', '7'),
    (160, 650, 310, 800, 'RIDE BELL', '8'),
    (320, 700, 470, 850, 'HITHAT CLOSE', '6'),
    (480, 750, 630, 900, 'CRASH', '9'),
    (640, 800, 790, 950, 'SNARE', '2'),
    (800, 850, 950, 1000, 'SNARE RIM', '3'),
    (960, 850, 1110, 1000, 'HIT HAT', '4'),
    (1120, 800, 1270, 950, 'HIT HAT OPEN', '5'),
    (1280, 750, 1430, 900, 'TOM HI', 'q'),
    (1440, 700, 1590, 850, 'TOM MID', 'w'),
    (1600, 650, 1750, 800, 'TOM LOW', 'e'),
    (1760, 600, 1900, 750, 'KICK', '1'),
]

recording = False

def start_recording():
    global recording
    recording = True
    stream.start_stream()

def stop_recording():
    global recording
    recording = False
    stream.stop_stream()

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = imutils.resize(frame, height=1400, width=1920)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lowred = np.array([131, 90, 106])
    highred = np.array([255, 255, 255])

    lowblue = np.array([40, 150, 116])
    highblue = np.array([255, 255, 255])

    red_mask = cv2.inRange(hsv, lowred, highred)
    blue_mask = cv2.inRange(hsv, lowblue, highblue)

    for (x1, y1, x2, y2, label, key) in positions:
        if label == 'RIDE':
            overlay_image(frame, ride_img, (x1, y1, x2, y2))
        elif label == 'RIDE BELL':
            overlay_image(frame, ride_bell_img, (x1, y1, x2, y2))
        elif label == 'HITHAT CLOSE':
            overlay_image(frame, hithat_close_img, (x1, y1, x2, y2))
        elif label == 'CRASH':
            overlay_image(frame, crash_img, (x1, y1, x2, y2))
        elif label == 'SNARE':
            overlay_image(frame, snare_img, (x1, y1, x2, y2))
        elif label == 'SNARE RIM':
            overlay_image(frame, snarerim_img, (x1, y1, x2, y2))
        elif label == 'HIT HAT':
            overlay_image(frame, hithat_img, (x1, y1, x2, y2))
        elif label == 'HIT HAT OPEN':
            overlay_image(frame, hithatopen_img, (x1, y1, x2, y2))
        elif label == 'TOM HI':
            overlay_image(frame, tomhi_img, (x1, y1, x2, y2))
        elif label == 'TOM MID':
            overlay_image(frame, tommid_img, (x1, y1, x2, y2))
        elif label == 'TOM LOW':
            overlay_image(frame, tomlow_img, (x1, y1, x2, y2))
        elif label == 'KICK':
            overlay_image(frame, kick_img, (x1, y1, x2, y2))

    def handle_mask(mask, frame, positions):
        global recording
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            pos = check_position(x, y, positions)
            if pos:
                x1, y1, x2, y2, label, key = pos
                if key in sounds:
                    sounds[key].play()
                if not recording:
                    start_recording()
                if label == 'RIDE':
                    overlay_image(frame, ride_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'RIDE BELL':
                    overlay_image(frame, ride_bell_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'HITHAT CLOSE':
                    overlay_image(frame, hithat_close_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'CRASH':
                    overlay_image(frame, crash_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'SNARE':
                    overlay_image(frame, snare_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'SNARE RIM':
                    overlay_image(frame, snarerim_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'HIT HAT':
                    overlay_image(frame, hithat_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'HIT HAT OPEN':
                    overlay_image(frame, hithatopen_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'TOM HI':
                    overlay_image(frame, tomhi_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'TOM MID':
                    overlay_image(frame, tommid_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'TOM LOW':
                    overlay_image(frame, tomlow_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                elif label == 'KICK':
                    overlay_image(frame, kick_img, (x1, y1, x1 + 155, y1 + 155), size=(155, 155))
                break

    handle_mask(red_mask, frame, positions)
    handle_mask(blue_mask, frame, positions)

    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        break

# Stop recording
stop_recording()

# Save the recorded frames as a wave file
stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

cap.release()
cv2.destroyAllWindows()


