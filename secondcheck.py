import cv2
import mediapipe as mp
import json
import os
import numpy as np
from pynput.keyboard import Controller

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize the keyboard controller
keyboard_controller = Controller()

# File to store gesture-to-key mappings
MAPPINGS_FILE = 'gesture_key_mappings.json'

# Dictionary to store gesture-to-key mappings
gesture_key_map = {}

# Define a recognition threshold
RECOGNITION_THRESHOLD = 0.5  # Adjust this value as needed

def load_mappings():
    """Load gesture-to-key mappings from a JSON file."""
    global gesture_key_map
    if os.path.exists(MAPPINGS_FILE):
        with open(MAPPINGS_FILE, 'r') as f:
            gesture_key_map = json.load(f)
            # Convert landmarks from lists to tuples
            for key, landmarks in gesture_key_map.items():
                gesture_key_map[key] = [tuple(lm) for lm in landmarks]
            print("Mappings loaded successfully.")
    else:
        print("No mappings file found. Starting fresh.")

def save_mappings():
    """Save gesture-to-key mappings to a JSON file."""
    with open(MAPPINGS_FILE, 'w') as f:
        # Convert landmarks from tuples to lists for JSON serialization
        json.dump({key: [list(lm) for lm in landmarks] for key, landmarks in gesture_key_map.items()}, f)
        print("Mappings saved successfully.")

def record_gesture(hand_landmarks):
    """Records the landmark positions of a hand gesture."""
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append((landmark.x, landmark.y, landmark.z))
    return landmarks

def compare_gestures(landmarks1, landmarks2):
    """Compares two sets of landmarks and returns a similarity score."""
    lm1 = np.array(landmarks1).flatten()
    lm2 = np.array(landmarks2).flatten()
    distance = np.linalg.norm(lm1.astype(np.float64) - lm2.astype(np.float64))
    return distance

def detect_gesture(hand_landmarks):
    """Detect and recognize gestures based on hand landmarks."""
    current_landmarks = record_gesture(hand_landmarks)
    
    min_distance = float('inf')
    detected_gesture = None
    
    for gesture, stored_landmarks in gesture_key_map.items():
        for stored_landmark_set in stored_landmarks:
            distance = compare_gestures(current_landmarks, stored_landmark_set)
            if distance < min_distance:
                min_distance = distance
                detected_gesture = gesture
    
    # Return detected gesture only if it is within the recognition threshold
    if min_distance > RECOGNITION_THRESHOLD:
        detected_gesture = None

    return detected_gesture

def map_gesture_to_key():
    """Capture a gesture and map it to a key."""
    print("Show the gesture you want to map and press 'Space' to capture.")
    cap = cv2.VideoCapture(0)

    captured_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                captured_landmarks = record_gesture(hand_landmarks)
        
        cv2.imshow("Capture Gesture", frame)

        if keyboard.is_pressed('space'):
            print("Gesture captured.")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            print("Gesture mapping cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()

    if captured_landmarks:
        key = input("Enter the key to map this gesture to (e.g., 'a', 'esc', 'space', 'enter'): ").strip().lower()
        if key in gesture_key_map:
            gesture_key_map[key].append(captured_landmarks)
        else:
            gesture_key_map[key] = [captured_landmarks]
        save_mappings()
        print(f"Gesture mapped to key '{key}'")

def trigger_key(gesture):
    """Simulate a key press based on the detected gesture."""
    if gesture:
        try:
            keyboard_controller.press(gesture)
            keyboard_controller.release(gesture)
            print(f"Triggered key '{gesture}'")
        except Exception as e:
            print(f"Error triggering key: {e}")

def capture_and_detect():
    """Capture video and detect gestures."""
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                gesture = detect_gesture(hand_landmarks)
                if gesture:  # Only trigger action if a valid gesture is detected
                    trigger_key(gesture)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

def list_gestures():
    """Lists all stored gestures and their corresponding keys."""
    print("Stored Gestures:")
    for gesture, key in gesture_key_map.items():
        print(f"- {gesture}: {key}")

def select_gesture():
    """Selects a stored gesture for editing."""
    list_gestures()
    gesture_to_edit = input("Enter the gesture name to edit: ")
    if gesture_to_edit in gesture_key_map:
        return gesture_to_edit
    else:
        print("Gesture not found.")
        return None

def edit_gesture(gesture):
    """Edits a stored gesture."""
    print(f"Editing gesture: {gesture}")
    cap = cv2.VideoCapture(0)

    captured_landmarks = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                captured_landmarks = record_gesture(hand_landmarks)
        
        cv2.imshow("Capture Gesture", frame)

        if keyboard.is_pressed('space'):
            print("Gesture captured.")
            break

        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            print("Gesture editing cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return
    
    cap.release()
    cv2.destroyAllWindows()

    if captured_landmarks:
        gesture_key_map[gesture].append(captured_landmarks)
        save_mappings()
        print(f"Gesture '{gesture}' updated successfully.")

def delete_gesture():
    """Deletes a stored gesture."""
    gesture_to_delete = select_gesture()
    if gesture_to_delete:
        confirm = input(f"Are you sure you want to delete the gesture '{gesture_to_delete}'? (yes/no): ").strip().lower()
        if confirm == 'yes':
            del gesture_key_map[gesture_to_delete]
            save_mappings()
            print(f"Gesture '{gesture_to_delete}' deleted successfully.")
        else:
            print("Deletion cancelled.")

def main_menu():
    """Main menu for the program."""
    load_mappings()  # Load existing mappings at the start

    while True:
        print("\n--- Gesture to Key Mapping ---")
        print("1. Map a gesture to a key")
        print("2. Start gesture detection")
        print("3. Show current mappings")
        print("4. Edit a gesture")
        print("5. Delete a gesture")
        print("6. Quit")
        choice = input("Enter your choice: ")

        if choice == '1':
            map_gesture_to_key()
        elif choice == '2':
            print("Starting gesture detection. Press ESC to stop.")
            capture_and_detect()
        elif choice == '3':
            list_gestures()
        elif choice == '4':
            gesture_to_edit = select_gesture()
            if gesture_to_edit:
                edit_gesture(gesture_to_edit)
        elif choice == '5':
            delete_gesture()
        elif choice == '6':
            print("Exiting.")
            save_mappings()  # Save mappings before exiting
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()
