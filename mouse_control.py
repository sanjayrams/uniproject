import pyautogui
import threading

class MouseControl:
    def __init__(self):
        self.lock = threading.Lock()

    def move_mouse(self, x, y):
        with self.lock:
            pyautogui.moveTo(x, y)

    def click(self):
        with self.lock:
            pyautogui.click()

    def right_click(self):
        with self.lock:
            pyautogui.rightClick()

    def double_click(self):
        with self.lock:
            pyautogui.doubleClick()

    def drag_mouse(self, x, y):
        with self.lock:
            pyautogui.dragTo(x, y)
