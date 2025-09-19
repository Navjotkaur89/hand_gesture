import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # We will only detect one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

def collect_data():
    """
    Captures hand landmarks from the webcam and saves them to a CSV file.
    The user specifies the gesture label, and the script appends data to 'gestures.csv'.
    """
    # Get the gesture name from the user
    gesture_label = input("Enter the name for the gesture you are about to record: ").strip().lower()
    if not gesture_label:
        print("Gesture name cannot be empty. Exiting.")
        return

    print(f"Recording data for gesture: '{gesture_label}'. Press 'q' to stop.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # CSV file setup
    csv_file_name = 'gestures.csv'
    file_exists = os.path.isfile(csv_file_name)

    with open(csv_file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file is new
        if not file_exists:
            header = ['label']
            for i in range(21):  # 21 landmarks
                header += [f'x{i}', f'y{i}', f'z{i}']
            writer.writerow(header)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display
            # and convert the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks and write to CSV
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    writer.writerow([gesture_label] + landmarks)

            # Show the image
            cv2.imshow('Hand Gesture Data Collection', image)

            # Exit on 'q' key
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    print(f"Data collection for '{gesture_label}' finished.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    collect_data()
