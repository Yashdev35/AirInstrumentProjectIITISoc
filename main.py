import cv2
import mediapipe as mp
import time
from GuitarCode.GuitarMain import VirGuitar

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Classes for different instruments
class Guitar:
    def play(self):
        print("Playing Guitar...")

class Piano:
    def play(self):
        print("Playing Piano...")

class Drum:
    def play(self):
        print("Playing Drum...")

def count_fingers(hand_landmarks):
    # Count the number of fingers raised
    finger_tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                   mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    finger_fold_status = []

    for tip in finger_tips:
        tip_id = tip.value
        # Check if finger tip is above the corresponding DIP joint
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            finger_fold_status.append(True)
        else:
            finger_fold_status.append(False)

    return sum(finger_fold_status)

def main():
    cap = cv2.VideoCapture(0)

    selected_instrument = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Count the number of raised fingers
                fingers_count = count_fingers(hand_landmarks)

                # Map the number of fingers to an instrument
                if fingers_count == 1:
                    selected_instrument = "Guitar"
                elif fingers_count == 2:
                    selected_instrument = "Piano"
                elif fingers_count == 3:
                    selected_instrument = "Drum"
                else:
                    selected_instrument = None

                if selected_instrument:
                    cv2.putText(frame, f"Instrument: {selected_instrument}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Instrument Selector', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if selected_instrument:
        confirm = input(f"Confirm playing {selected_instrument}? (y/n): ")
        cv2.text(frame, "Confirm playing {selected_instrument}? (y/n): ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if confirm.lower() == 'y':
            if selected_instrument == "Guitar":
                VirGuitar()
            elif selected_instrument == "Piano":
                Piano().play()
            elif selected_instrument == "Drum":
                Drum().play()

if __name__ == "__main__":
    main()
