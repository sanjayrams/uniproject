import cv2
import mediapipe as mp
import json
import os
import threading
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class GestureRecognition:
    def __init__(self, config):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.gesture_mappings = {}
        self.load_gestures()
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.train_model()

    def load_gestures(self):
        if os.path.exists("gesture_mappings.json"):
            with open("gesture_mappings.json", "r") as f:
                self.gesture_mappings = json.load(f)

    def save_gestures(self):
        with open("gesture_mappings.json", "w") as f:
            json.dump(self.gesture_mappings, f)

    def capture_gesture(self, gesture_name):
        cap = cv2.VideoCapture(self.config['camera_index'])
        print("Recording gesture... Move your hand and press 'q' to finish.")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('Record Gesture', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb.tolist())
        cap.release()
        cv2.destroyAllWindows()
        self.gesture_mappings[gesture_name] = frames
        self.train_model()

    def extract_features(self, hand_landmarks):
        return np.array([[(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]]).flatten()

    def train_model(self):
        X, y = [], []
        for gesture, frames in self.gesture_mappings.items():
            for frame in frames:
                result = self.hands.process(np.array(frame))
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        X.append(self.extract_features(hand_landmarks))
                        y.append(gesture)
        if X and y:
            self.model.fit(X, y)

    def recognize_gesture(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                features = self.extract_features(hand_landmarks)
                gesture = self.model.predict([features])
                return gesture[0]
        return None

    def delete_gesture(self, gesture_name):
        if gesture_name in self.gesture_mappings:
            del self.gesture_mappings[gesture_name]
            self.save_gestures()

    def edit_gesture(self, gesture_name):
        if gesture_name in self.gesture_mappings:
            self.capture_gesture(gesture_name)
            self.save_gestures()
