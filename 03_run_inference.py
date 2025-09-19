import cv2
import mediapipe as mp
import pickle
import numpy as np

def run_inference():
    """
    Loads the trained model and performs real-time gesture recognition
    using the webcam.
    """
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

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting real-time gesture recognition. Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find hands
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                # Extract landmarks for model prediction
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Make a prediction
                prediction = model.predict([landmarks])
                predicted_gesture = prediction[0]
                
                # Get prediction probability
                proba = model.predict_proba([landmarks])
                confidence = np.max(proba)

                # Display the prediction on the screen
                text = f"{predicted_gesture} ({confidence*100:.2f}%)"
                
                # Get the bounding box of the hand to position the text
                h, w, _ = image.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                text_x = int(min(x_coords) * w)
                text_y = int(min(y_coords) * h) - 10

                cv2.putText(image, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image
        cv2.imshow('Hand Gesture Recognition', image)

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == '__main__':
    run_inference()
