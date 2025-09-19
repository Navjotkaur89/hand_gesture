import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyautogui
import math
import time

# --- SETTINGS ---
# Set a lower confidence threshold for actions to avoid accidental triggers
ACTION_CONFIDENCE_THRESHOLD = 0.95 
# Distance threshold for triggering a click (in normalized coordinates)
CLICK_DISTANCE_THRESHOLD = 0.04 
# Cooldown in seconds to prevent rapid-fire clicks
CLICK_COOLDOWN = 0.5

def run_gesture_control():
    """
    Loads the trained model and performs real-time gesture recognition
    to control the mouse and system volume.
    """
    # Disable PyAutoGUI's fail-safe to allow control near screen corners
    pyautogui.FAILSAFE = False

    # Load the trained model
    model_filename = 'gesture_model.pkl'
    try:
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Please run the training script first.")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    # Get screen dimensions
    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting gesture control. Move hand to control mouse. Press 'q' to quit.")
    
    last_click_time = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- 1. MOUSE MOVEMENT ---
                # Use index finger tip (landmark 8) to control the cursor
                index_finger_tip = hand_landmarks.landmark[8]
                # The landmarks are normalized (0.0 to 1.0), so we scale them to the screen size
                mouse_x = int(index_finger_tip.x * screen_width)
                mouse_y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(mouse_x, mouse_y)

                # --- 2. CLICK ACTION ---
                # Use distance between thumb tip (4) and index finger tip (8)
                thumb_tip = hand_landmarks.landmark[4]
                distance = math.hypot(index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y)

                # Check for click gesture and cooldown
                current_time = time.time()
                if distance < CLICK_DISTANCE_THRESHOLD and (current_time - last_click_time) > CLICK_COOLDOWN:
                    pyautogui.click()
                    print("Click!")
                    last_click_time = current_time
                    # Draw a circle to indicate a click
                    cv2.circle(image, (int(index_finger_tip.x * image.shape[1]), int(index_finger_tip.y * image.shape[0])), 15, (0, 255, 0), 3)


                # --- 3. GESTURE-BASED ACTIONS (e.g., Volume Control) ---
                landmarks = [lm for lm in hand_landmarks.landmark]
                flat_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
                
                prediction = model.predict([flat_landmarks])
                predicted_gesture = prediction[0]
                proba = model.predict_proba([flat_landmarks])
                confidence = np.max(proba)

                # Display the prediction on the screen
                text = f"{predicted_gesture} ({confidence*100:.2f}%)"
                cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Perform action only if confidence is high
                if confidence > ACTION_CONFIDENCE_THRESHOLD:
                    # NOTE: Add your own gestures and actions here!
                    # Make sure you have collected data for 'thumbs_up' and 'peace'
                    if predicted_gesture == 'thumbs_up':
                        pyautogui.press('volumeup')
                        print("Volume Up")
                    elif predicted_gesture == 'peace':
                        pyautogui.press('volumedown')
                        print("Volume Down")
                    elif predicted_gesture == 'fist':
                        pyautogui.press('space') # Example: Pause/Play video
                        print("Play/Pause (Spacebar)")


        cv2.imshow('Gesture Control', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture control stopped.")

if __name__ == '__main__':
    run_gesture_control()
