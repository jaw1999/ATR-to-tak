from ultralytics import YOLO
import cv2
import time
import signal
import sys

def signal_handler(sig, frame):
    print('Ctrl+C pressed. Cleaning up...')
    if 'cap' in globals():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def main():
    global cap
    signal.signal(signal.SIGINT, signal_handler)
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    window_width = 640
    window_height = 480
    cv2.resizeWindow('Object Detection', window_width, window_height)
    print("Window created...")
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    model.verbose = False
    print("Model loaded successfully!")
    try:
        print("Starting video feed... Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            results = model(frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('Object Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('f'):
                current_property = cv2.getWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if current_property == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
            time.sleep(0.01)
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        for i in range(1,5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main() 