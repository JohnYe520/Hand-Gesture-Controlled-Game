import cv2, os, time

# gesture key mapping
GESTURE_KEYS = {
    'u': 'up',
    'd': 'down',
    'l': 'left',
    'r': 'right',
    's': 'stop',
    'z': 'zero',
    'n': 'none'  # background / no hand
}

SAVE_DIR = "dataset_retrain"
os.makedirs(SAVE_DIR, exist_ok=True)

# start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()

roi_size = 400
print("[INFO] Press keys (u/d/l/r/s/z/n) to save ROI for that gesture.")
print("[INFO] Press q to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    cx, cy = w//2, h//2
    x1, y1 = cx - roi_size//2, cy - roi_size//2
    x2, y2 = cx + roi_size//2, cy + roi_size//2

    # draw ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]
    cv2.imshow("Webcam", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif chr(key) in GESTURE_KEYS:
        gesture = GESTURE_KEYS[chr(key)]
        folder = os.path.join(SAVE_DIR, gesture)
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, f"{gesture}_{int(time.time()*1000)}.jpg")
        cv2.imwrite(filename, roi)
        print(f"[SAVED] {filename}")

cap.release()
cv2.destroyAllWindows()
