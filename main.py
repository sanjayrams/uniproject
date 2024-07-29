import json
import threading
import cv2
from gesture_recognition import GestureRecognition
from mouse_control import MouseControl

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def run_mouse_emulation(gesture_recognition, mouse_control):
    cap = cv2.VideoCapture(config['camera_index'])
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gesture = gesture_recognition.recognize_gesture(frame)
        if gesture == "move":
            coords = gesture_recognition.recognize_gesture(frame)
            if coords:
                x, y = coords
                mouse_control.move_mouse(x, y)
        elif gesture == "click":
            mouse_control.click()

        cv2.imshow('Mouse Emulation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def add_custom_gesture(gesture_recognition):
    gesture_name = input("Enter the name of the new gesture: ")
    print("Record the gesture...")
    gesture_recognition.capture_gesture(gesture_name)
    gesture_recognition.save_gestures()
    print(f"Gesture '{gesture_name}' added successfully.")

def edit_gesture_mapping(gesture_recognition):
    gesture_name = input("Enter the name of the gesture to edit: ")
    if gesture_name in gesture_recognition.gesture_mappings:
        print(f"Editing gesture '{gesture_name}'.")
        gesture_recognition.capture_gesture(gesture_name)
        gesture_recognition.save_gestures()
        print(f"Gesture '{gesture_name}' updated successfully.")
    else:
        print(f"Gesture '{gesture_name}' not found.")

def delete_gesture_mapping(gesture_recognition):
    gesture_name = input("Enter the name of the gesture to delete: ")
    gesture_recognition.delete_gesture(gesture_name)
    print(f"Gesture '{gesture_name}' deleted successfully.")

def main_menu():
    gesture_recognition = GestureRecognition(config)
    mouse_control = MouseControl()

    while True:
        print("\nMain Menu:")
        print("1. Run Mouse Emulation Program")
        print("2. Add Custom Gesture")
        print("3. Edit Existing Gesture Mapping")
        print("4. Delete Custom Gesture Mapping")
        print("5. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            emulation_thread = threading.Thread(target=run_mouse_emulation, args=(gesture_recognition, mouse_control))
            emulation_thread.start()
            emulation_thread.join()
        elif choice == '2':
            add_custom_gesture(gesture_recognition)
        elif choice == '3':
            edit_gesture_mapping(gesture_recognition)
        elif choice == '4':
            delete_gesture_mapping(gesture_recognition)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    config = load_config()
    main_menu()
