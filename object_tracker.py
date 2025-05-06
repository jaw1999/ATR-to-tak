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
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize the webcam
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Create window first
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    # Set a smaller window size
    window_width = 640  # Default camera resolution width
    window_height = 480  # Default camera resolution height
    cv2.resizeWindow('Object Detection', window_width, window_height)
    print("Window created...")
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    model.verbose = False  # Disable speed printouts
    print("Model loaded successfully!")
        
    try:
        print("Starting video feed... Press 'q' to quit")
        
        while True:
            # Read a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # No need to resize frame - use native camera resolution
            
            # Run detection
            results = model(frame)
            
            # Process the results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Get class name and confidence
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Add label
                    label = f'{class_name} {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Object Detection', frame)
            
            # Break the loop if 'q' is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('f'):  # Press 'f' to toggle fullscreen
                # Toggle between normal and fullscreen
                current_property = cv2.getWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN,
                                    cv2.WINDOW_NORMAL if current_property == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
    
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        # Additional cleanup attempts
        for i in range(1,5):
            cv2.waitKey(1)

if __name__ == "__main__":
    main() 