import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create data directory
DATA_DIR = 'sign_data'
GESTURES = ['hello', 'thank_you', 'yes', 'no', 'please', 'goodbye', 'love', 'peace']

def create_directories():
    """Create folders for each gesture"""
    for gesture in GESTURES:
        path = os.path.join(DATA_DIR, gesture)
        os.makedirs(path, exist_ok=True)
    print(f"✅ Created directories in '{DATA_DIR}'")

def collect_gestures():
    """Capture hand images from webcam"""
    cap = cv2.VideoCapture(0)

    current_gesture = 0
    image_count = 0
    collecting = False
    max_images = 500  # Images per gesture

    print("\n" + "="*50)
    print("👐 ES RELIVE Data Collection")
    print("="*50)
    print("\nInstructions:")
    print("  • Show hand gesture in camera")
    print("  • Press 'SPACE' to start/stop collecting")
    print("  • Press '1-8' to change gesture")
    print("  • Press 'Q' to quit")
    print("="*50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and convert color
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hands
        results = hands.process(rgb)

        # Draw UI
        cv2.rectangle(frame, (0, 0), (640, 80), (50, 50, 50), -1)
        cv2.putText(frame, f"Gesture: {GESTURES[current_gesture].upper()}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Images: {image_count}/{max_images}",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        status = "RECORDING" if collecting else "PAUSED"
        color = (0, 255, 0) if collecting else (100, 100, 100)
        cv2.putText(frame, status, (450, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Save image if hand detected and collecting
        if results.multi_hand_landmarks and collecting:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Save image
                if image_count < max_images:
                    img_path = os.path.join(DATA_DIR, GESTURES[current_gesture],
                                           f"{GESTURES[current_gesture]}_{image_count}.jpg")
                    cv2.imwrite(img_path, frame)
                    image_count += 1

                    if image_count % 50 == 0:
                        print(f"  📸 {GESTURES[current_gesture]}: {image_count} images")

        cv2.imshow('ES RELIVE Data Collection', frame)

        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            collecting = not collecting
            if collecting:
                image_count = 0
                print(f"\n▶ Started collecting: {GESTURES[current_gesture]}")
            else:
                print(f"⏸ Paused: {image_count} images saved")
        elif key == ord('q'):
            break
        elif key >= ord('1') and key <= ord('8'):
            idx = key - ord('1')
            if idx < len(GESTURES):
                current_gesture = idx
                image_count = 0
                collecting = False
                print(f"\n🔄 Switched to: {GESTURES[current_gesture]}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Data collection complete!")

if __name__ == "__main__":
    create_directories()
    print("\n📁 Data will be saved to: " + DATA_DIR)
    print("🧤 Showing hand in frame and press SPACE to start\n")
    collect_gestures()