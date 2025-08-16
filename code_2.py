import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import time
from collections import deque, Counter
import math


class SignLanguageDetector:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        # Set max_num_hands to 1 as the new sign logic focuses on single-hand gestures
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Focusing on single hand for the provided signs
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0  # Optimized for speed
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Sign gesture patterns and descriptions for the UI
        self.sign_descriptions = {
            'HELLO': "All fingers extended (open hand).",
            'OKAY': "Thumb and index finger forming a circle, others extended.",
            'BYE': "Index and middle fingers extended (V-shape, like Peace sign).",
            'GOOD': "Thumb extended, other fingers folded (Thumbs up).",
            'POINT': "Index finger extended, other fingers folded.",
            'STOP': "All fingers folded (closed fist).",
            'I LOVE YOU': "Thumb, index, and pinky fingers extended, middle and ring fingers folded."
        }

        # Sign detection variables for stability and display
        self.current_sign = "None"
        # Longer window for stability (was 20)
        self.sign_history = deque(maxlen=25)
        self.sign_confidence = 0.0  # Fixed confidence for rule-based detection
        self.is_running = False
        self.last_added_sign = ""  # To prevent repeated additions of the same sign

        # Number of consistent frames needed for a sign to be confirmed
        self.min_stable_frames = 10  # was 15, a bit friendlier

        # ---- Tunables for robustness ----
        self.OKAY_DIST = 0.08  # normalized distance threshold for "OKAY"
        self.Y_TOL = 0.02      # small Y tolerance for finger extension

        # Initialize camera
        self.cap = None
        self.initialize_camera()

        # UI setup
        self.root = tk.Tk()
        self.setup_ui()

    def initialize_camera(self):
        """Initialize camera with error handling and preferred backend."""
        try:
            # Try specific backend for potentially better performance on Windows
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default if CAP_DSHOW fails or for other OS
                print(
                    "CAP_DSHOW failed or not applicable, trying default camera indices.")
                for i in range(0, 4):  # Iterate through common camera indices
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        break

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
                # Keep buffer small so we always get the freshest frame
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                print("Camera initialized successfully.")
            else:
                print("Warning: Could not initialize camera.")
        except Exception as e:
            print(f"Camera initialization error: {e}")
            self.cap = None

    def setup_ui(self):
        """Setup the user interface elements."""
        self.root.title("Sign Language Detector - Communication Aid")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        self.root.resizable(True, True)

        # Header Frame
        header_frame = tk.Frame(self.root, bg='#34495e', height=70)
        header_frame.pack(fill='x', padx=10, pady=5)
        header_frame.pack_propagate(False)

        title_label = tk.Label(
            header_frame,
            text="🤟 Sign Language Detector",
            font=('Arial', 20, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        title_label.pack(pady=20)

        # Main Container for Left and Right Panels
        main_container = tk.Frame(self.root, bg='#2c3e50')
        main_container.pack(fill='both', expand=True, padx=10, pady=5)

        # Left Panel (Camera Feed and Controls)
        left_panel = tk.Frame(main_container, bg='#34495e', width=600)
        left_panel.pack(side='left', fill='both', expand=True, padx=5)

        camera_frame = tk.LabelFrame(
            left_panel,
            text="Camera Feed",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e',
            relief='raised',
            bd=2
        )
        camera_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.camera_label = tk.Label(
            camera_frame,
            text="Camera will appear here\nClick 'Start Detection' to begin",
            font=('Arial', 14),
            fg='#bdc3c7',
            bg='#2c3e50',
            justify='center'
        )
        self.camera_label.pack(fill='both', expand=True, padx=10, pady=10)

        control_frame = tk.Frame(left_panel, bg='#34495e', height=80)
        control_frame.pack(fill='x', padx=5, pady=5)
        control_frame.pack_propagate(False)

        button_style = {'font': ('Arial', 11, 'bold'),
                        'width': 12, 'height': 2}

        self.start_button = tk.Button(
            control_frame,
            text="▶ Start",
            command=self.start_detection,
            bg='#27ae60',
            fg='white',
            **button_style
        )
        self.start_button.pack(side='left', padx=5, pady=10)

        self.stop_button = tk.Button(
            control_frame,
            text="⏸ Stop",
            command=self.stop_detection,
            bg='#e74c3c',
            fg='white',
            state='disabled',
            **button_style
        )
        self.stop_button.pack(side='left', padx=5, pady=10)

        self.clear_button = tk.Button(
            control_frame,
            text="🗑 Clear",
            command=self.clear_text,
            bg='#f39c12',
            fg='white',
            **button_style
        )
        self.clear_button.pack(side='left', padx=5, pady=10)

        # Right Panel (Detection Info, Detected Text, Instructions)
        right_panel = tk.Frame(main_container, bg='#34495e', width=550)
        right_panel.pack(side='right', fill='both', expand=True, padx=5)

        detection_frame = tk.LabelFrame(
            right_panel,
            text="Current Detection",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        detection_frame.pack(fill='x', padx=5, pady=5)

        self.current_sign_label = tk.Label(
            detection_frame,
            text="None",
            font=('Arial', 32, 'bold'),
            fg='#3498db',
            bg='#34495e'
        )
        self.current_sign_label.pack(pady=10)

        self.confidence_label = tk.Label(
            detection_frame,
            text="Confidence: 0%",
            font=('Arial', 12),
            fg='#95a5a6',
            bg='#34495e'
        )
        self.confidence_label.pack(pady=2)

        self.status_label = tk.Label(
            detection_frame,
            text="● Stopped",
            font=('Arial', 10, 'bold'),
            fg='#e74c3c',
            bg='#34495e'
        )
        self.status_label.pack(pady=5)

        output_frame = tk.LabelFrame(
            right_panel,
            text="Detected Text",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        output_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.text_display = scrolledtext.ScrolledText(
            output_frame,
            height=8,
            font=('Arial', 14, 'bold'),
            wrap=tk.WORD,
            bg='#ecf0f1',
            fg='#2c3e50',
            insertbackground='#2c3e50'
        )
        self.text_display.pack(fill='both', expand=True, padx=10, pady=10)

        instructions_frame = tk.LabelFrame(
            right_panel,
            text="Sign Instructions",
            font=('Arial', 12, 'bold'),
            fg='#ecf0f1',
            bg='#34495e'
        )
        instructions_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.instruction_text = scrolledtext.ScrolledText(
            instructions_frame,
            height=10,
            font=('Arial', 10),
            wrap=tk.WORD,
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        self.instruction_text.pack(fill='both', expand=True, padx=10, pady=10)

        self.load_instructions()

    def load_instructions(self):
        """Load sign instructions into the text widget based on self.sign_descriptions."""
        instructions = "📋 HOW TO MAKE SIGNS (Simplified Rules):\n\n"
        instructions += "NOTE: This detector uses basic rule-based recognition for a limited set of signs. Real sign language is much more complex!\n\n"
        for sign, description in self.sign_descriptions.items():
            instructions += f"• {sign}: {description}\n"

        instructions += "\n💡 TIPS FOR BEST RESULTS:\n"
        instructions += "• Ensure good lighting and a clear background.\n"
        instructions += "• Position your hand clearly in the camera frame.\n"
        instructions += "• Make gestures slowly and hold them steady for detection.\n"

        self.instruction_text.insert(tk.END, instructions)
        self.instruction_text.config(state='disabled')

    def calculate_distance(self, p1, p2):
        """Euclidean distance in normalized coordinates (x,y)."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_status(self, landmarks, handedness: str):
        """
        Determine which fingers are extended (1) or folded (0) based on tip vs. PIP joint Y-coordinates.
        Thumb is handled with handedness.
        """
        tip_ids = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        pip_ids = [3, 6, 10, 14, 18]   # PIP joints

        # Thumb: Compare X-coordinate of tip to PIP joint depending on handedness.
        is_right = handedness.lower().startswith("right")
        thumb_extended = (landmarks[tip_ids[0]].x > landmarks[pip_ids[0]].x) if is_right \
            else (landmarks[tip_ids[0]].x < landmarks[pip_ids[0]].x)

        fingers = [1 if thumb_extended else 0]

        # Other fingers (Index, Middle, Ring, Pinky): Tip above PIP (smaller y) with tolerance
        for tip, pip in zip(tip_ids[1:], pip_ids[1:]):
            fingers.append(1 if landmarks[tip].y < (
                landmarks[pip].y - self.Y_TOL) else 0)

        return fingers

    def recognize_sign(self, hand_data_list):
        """
        Single-hand rule-based recognition with consistent if/elif priority.
        """
        if not hand_data_list:
            return 'No Hands Detected', 0.0

        hand = hand_data_list[0]
        hand_landmarks = hand['landmarks'].landmark
        handedness = hand.get('label', 'Right')

        fingers = self.get_finger_status(hand_landmarks, handedness)
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)

        # Priority order:
        # 1) OKAY (index touching thumb) with the rest extended
        if distance < self.OKAY_DIST and fingers[2:] == [1, 1, 1]:
            return "OKAY", 0.9

        # 2) HELLO (all open)
        elif fingers == [1, 1, 1, 1, 1]:
            return "HELLO", 0.9

        # 3) BYE (peace / V sign)
        elif fingers == [0, 1, 1, 0, 0]:
            return "BYE", 0.8

        # 4) GOOD (thumbs up)
        elif fingers == [1, 0, 0, 0, 0]:
            return "GOOD", 0.8

        # 5) POINT (index only)
        elif fingers == [0, 1, 0, 0, 0]:
            return "POINT", 0.8

        # 6) STOP (fist)
        elif fingers == [0, 0, 0, 0, 0]:
            return "STOP", 0.8

        # 7) I LOVE YOU (thumb, index, pinky)
        elif fingers == [1, 1, 0, 0, 1]:
            return "I LOVE YOU", 0.9

        return "UNKNOWN", 0.3

    def update_camera_feed(self):
        """Main camera processing loop, runs in a separate thread."""
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.1)  # Wait if camera not ready
                continue

            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)  # Flip for mirror effect

            # OPTIMIZATION: Resize frame BEFORE processing with MediaPipe
            # Smaller frames significantly reduce processing time.
            processing_frame = cv2.resize(
                frame, (320, 240), interpolation=cv2.INTER_AREA)

            frame_rgb_for_mediapipe = cv2.cvtColor(
                processing_frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb_for_mediapipe)

            detected_hands_data = []

            if results.multi_hand_landmarks:
                # Guard for multi_handedness possibly missing on some frames
                handed_list = getattr(results, "multi_handedness", [])
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    label = "Right"
                    if handed_list and idx < len(handed_list):
                        label = handed_list[idx].classification[0].label

                    detected_hands_data.append(
                        {'landmarks': hand_landmarks, 'label': label})

                    # Draw landmarks on the ORIGINAL sized 'frame'
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(
                            color=(255, 255, 0), thickness=2)
                    )

            # Recognize the sign using the new logic
            sign, confidence = self.recognize_sign(detected_hands_data)

            # Add current frame's detection to history
            self.sign_history.append(sign)

            # Stabilization logic: Confirm a sign only if it's consistent over several frames
            if len(self.sign_history) >= self.min_stable_frames:
                sign_counts = Counter(self.sign_history)
                most_common_sign, count = sign_counts.most_common(1)[0]

                if count >= self.min_stable_frames * 0.7 and \
                   most_common_sign not in ['UNKNOWN', 'No Hands Detected']:
                    # Add to text display only if it's a new, confirmed sign
                    if most_common_sign != self.last_added_sign:
                        self.current_sign = most_common_sign
                        self.sign_confidence = confidence
                        self.add_sign_to_text(self.current_sign)
                        self.last_added_sign = self.current_sign
                        self.sign_history.clear()  # Reset history after adding a stable sign
                    else:
                        # If same sign is stable again, just update current display, don't re-add to text
                        self.current_sign = most_common_sign
                        self.sign_confidence = confidence
                else:
                    # Not stable enough or it's an unknown/no hand state
                    self.current_sign = "Analyzing..."
                    self.sign_confidence = 0.0
            else:
                self.current_sign = "Collecting data..."  # Building up history buffer
                self.sign_confidence = 0.0

            # If no hands are detected, reset state
            if not detected_hands_data:
                self.sign_history.clear()
                self.last_added_sign = ""
                self.current_sign = "None"
                self.sign_confidence = 0.0

            # Convert frame for Tkinter display
            frame_rgb_for_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb_for_display)
            # Resize for UI display
            img = img.resize((560, 420), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image=img)

            # Schedule UI update on the main Tkinter thread
            self.root.after(0, self.update_display, photo)

            # Tiny yield to avoid pegging CPU and improve responsiveness
            time.sleep(0.001)

    def update_display(self, photo):
        """Updates the Tkinter UI elements with camera feed and detection results."""
        try:
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo  # Keep a reference to prevent garbage collection

            self.current_sign_label.config(text=self.current_sign)
            self.confidence_label.config(
                text=f"Confidence: {self.sign_confidence:.0%}"
            )

            if self.is_running:
                self.status_label.config(text="● Running", fg='#27ae60')
            else:
                self.status_label.config(text="● Stopped", fg='#e74c3c')

        except Exception as e:
            # This can happen if Tkinter window is closed while thread is still trying to update
            print(f"Display update error: {e}")

    def add_sign_to_text(self, sign):
        """Adds a confirmed detected sign to the scrolled text display."""
        try:
            current_text = self.text_display.get(1.0, tk.END).strip()

            # Add a space if the previous text doesn't end with one
            if current_text and not current_text.endswith(" "):
                self.text_display.insert(tk.END, " ")

            self.text_display.insert(tk.END, sign)
            # Scroll to the end to show latest text
            self.text_display.see(tk.END)

        except Exception as e:
            print(f"Text update error: {e}")

    def start_detection(self):
        """Starts the sign detection process by launching the camera thread."""
        if self.cap is None or not self.cap.isOpened():
            messagebox.showerror(
                "Error", "Camera not available or could not be opened! Please check camera connection.")
            return

        if not self.is_running:
            self.is_running = True
            # Reset detection state for a fresh start
            self.last_added_sign = ""
            self.sign_history.clear()

            # Start camera feed processing in a separate thread to keep UI responsive
            self.camera_thread = threading.Thread(
                target=self.update_camera_feed)
            self.camera_thread.daemon = True  # Allows thread to close when main program exits
            self.camera_thread.start()

            # Update UI button states and status label
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_label.config(text="● Running", fg='#27ae60')
            print("Detection started.")

    def stop_detection(self):
        """Stops the sign detection process."""
        self.is_running = False

        # Wait briefly for the camera thread to finish its current loop iteration
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=0.5)

        # Update UI elements to reflect stopped state
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        self.camera_label.configure(image='')
        self.camera_label.config(
            text="Camera stopped\nClick 'Start Detection' to resume"
        )
        self.status_label.config(text="● Stopped", fg='#e74c3c')
        self.current_sign_label.config(text="None")
        self.confidence_label.config(text="Confidence: 0%")

        # Clear detection buffers
        self.sign_history.clear()
        self.last_added_sign = ""
        print("Detection stopped.")

    def clear_text(self):
        """Clears the detected text display area."""
        self.text_display.delete(1.0, tk.END)
        self.last_added_sign = ""  # Reset last added sign to allow re-detection

    def run(self):
        """Starts the Tkinter main loop."""
        # Ensure proper cleanup when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Show camera warning if not initialized
        if self.cap is None or not self.cap.isOpened():
            messagebox.showwarning(
                "Camera Warning",
                "No camera detected or could not be opened. Please ensure your camera is connected and not in use by another application."
            )

        self.root.mainloop()

    def on_closing(self):
        """Handles application shutdown, releasing camera and destroying windows."""
        self.is_running = False  # Signal the camera thread to stop

        # Give the camera thread a moment to finish its loop and exit
        if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.0)
            if self.camera_thread.is_alive():
                print("Warning: Camera thread did not terminate cleanly.")

        if self.cap:
            self.cap.release()  # Release camera resource
            print("Camera released.")

        # Close mediapipe graph cleanly
        if hasattr(self, "hands") and self.hands:
            try:
                self.hands.close()
            except Exception:
                pass

        cv2.destroyAllWindows()  # Close any OpenCV windows
        print("OpenCV windows destroyed.")

        self.root.quit()  # End Tkinter main loop
        self.root.destroy()  # Destroy Tkinter window
        print("Tkinter app destroyed.")


def main():
    """Main function to run the application, including package checks."""
    print("🤟 Starting Sign Language Detector...")
    print("📋 This program helps with sign language communication.")
    print("📦 Required packages: opencv-python, mediapipe, pillow, numpy.")

    try:
        # Check if required packages are installed
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        from PIL import Image, ImageTk  # noqa: F401  # Pillow for Tkinter image handling
        print("✅ All required packages found.")

        app = SignLanguageDetector()
        app.run()

    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📥 Please install required packages:")
        print("pip install opencv-python mediapipe pillow numpy")

    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging


if __name__ == "__main__":
    main()
