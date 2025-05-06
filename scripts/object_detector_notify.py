from ultralytics import YOLO
import cv2
import time
import signal
import sys
import socket
import xml.etree.ElementTree as ET
from datetime import datetime
import tkinter as tk
from tkinter import ttk
import os

# Suppress OpenCV warnings
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

class DetectionConfig:
    def __init__(self):
        self.selected_objects = set()
        self.confidence_threshold = 0.5
        # Just load model with verbose=False
        self.model = YOLO('yolov8n.pt', verbose=False)
        self.detected_objects = set()
        self.latitude = 0.0
        self.longitude = 0.0
        self.cot_host = '127.0.0.1'
        self.cot_port = 6969
        self.callsign = "Detection1"  # Default name
        self.camera_source = 0
        self.show_video = True
        self.reset_timer = 0  # seconds between resets
        self.last_reset = time.time()  # track last reset time
        
        # Create UI window
        self.root = tk.Tk()
        self.root.title("Object Detection Settings")
        
        # Create frame for location inputs
        loc_frame = ttk.Frame(self.root)
        loc_frame.pack(padx=10, pady=5, fill='x')
        
        # Add latitude input
        ttk.Label(loc_frame, text="Latitude:").pack()
        self.lat_entry = ttk.Entry(loc_frame)
        self.lat_entry.pack()
        self.lat_entry.insert(0, "0.0")
        
        # Add longitude input
        ttk.Label(loc_frame, text="Longitude:").pack()
        self.lon_entry = ttk.Entry(loc_frame)
        self.lon_entry.pack()
        self.lon_entry.insert(0, "0.0")
        
        # Create frame for CoT destination
        cot_frame = ttk.Frame(self.root)
        cot_frame.pack(padx=10, pady=5, fill='x')
        
        # Add IP input
        ttk.Label(cot_frame, text="Destination IP:").pack()
        self.ip_entry = ttk.Entry(cot_frame)
        self.ip_entry.pack()
        self.ip_entry.insert(0, "127.0.0.1")
        
        # Add Port input
        ttk.Label(cot_frame, text="Destination Port:").pack()
        self.port_entry = ttk.Entry(cot_frame)
        self.port_entry.pack()
        self.port_entry.insert(0, "6969")
        
        # Create frame for name input
        name_frame = ttk.Frame(self.root)
        name_frame.pack(padx=10, pady=5, fill='x')
        
        # Add callsign input
        ttk.Label(name_frame, text="Detection Name:").pack()
        self.name_entry = ttk.Entry(name_frame)
        self.name_entry.pack()
        self.name_entry.insert(0, "Detection1")
        
        # Add confidence threshold slider
        ttk.Label(self.root, text="Confidence Threshold:").pack()
        self.conf_slider = ttk.Scale(
            self.root, 
            from_=0.0, 
            to=1.0, 
            value=0.5,
            orient='horizontal',
            command=self.update_confidence
        )
        self.conf_slider.pack(padx=10, pady=5)
        
        # Add object selection listbox
        ttk.Label(self.root, text="Select Objects to Detect:").pack()
        self.listbox = tk.Listbox(self.root, selectmode='multiple', height=10)
        self.listbox.pack(padx=10, pady=5)
        
        # Add Reset button
        ttk.Button(self.root, text="Reset Detections", command=self.reset_detections).pack(pady=5)
        
        # Populate listbox with available classes
        for idx, name in self.model.names.items():
            self.listbox.insert(tk.END, name)
        
        # Add Start button
        ttk.Button(self.root, text="Start Detection", command=self.start_detection).pack(pady=10)
        
        self.ready_to_start = False
    
    def update_confidence(self, value):
        self.confidence_threshold = float(value)
    
    def reset_detections(self):
        self.detected_objects.clear()
        print("Reset detection history")
    
    def start_detection(self):
        try:
            self.latitude = float(self.lat_entry.get())
            self.longitude = float(self.lon_entry.get())
            self.cot_port = int(self.port_entry.get())
            self.cot_host = self.ip_entry.get()
            self.callsign = self.name_entry.get()
            if not self.callsign:  # If empty, use default
                self.callsign = "Detection1"
        except ValueError:
            print("Invalid input values. Using defaults.")
            self.latitude = 0.0
            self.longitude = 0.0
            self.cot_host = '127.0.0.1'
            self.cot_port = 6969
            self.callsign = "Detection1"
            
        selected_indices = self.listbox.curselection()
        self.selected_objects = {self.listbox.get(i) for i in selected_indices}
        print(f"Selected objects: {self.selected_objects}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Location: {self.latitude}, {self.longitude}")
        self.ready_to_start = True
        self.root.quit()

def create_cot_message(class_name, confidence, lat, lon, callsign):
    """Create a CoT XML message for detection"""
    xml_header = '<?xml version="1.0" encoding="UTF-8"?>'
    
    root = ET.Element("event")
    root.set("version", "2.0")
    root.set("type", "b-d-A")  # Detection type
    root.set("uid", f"{callsign}-{int(time.time())}")
    root.set("time", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    root.set("start", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z"))
    root.set("stale", "2025-04-20T02:36:05.000Z")
    root.set("how", "m-g")

    point = ET.SubElement(root, "point")
    point.set("lat", f"{lat:.4f}")
    point.set("lon", f"{lon:.4f}")
    point.set("hae", "0.0")
    point.set("ce", "9999999.0")
    point.set("le", "9999999.0")

    detail = ET.SubElement(root, "detail")
    detail.set("class", class_name)
    detail.set("confidence", f"{confidence:.2f}")
    detail.set("callsign", callsign)
    detail.set("type", "Detection")  # Explicitly mark as detection

    message = f"{xml_header}\n" + ET.tostring(root, encoding='unicode', xml_declaration=False)
    return message

def signal_handler(sig, frame):
    print('Ctrl+C pressed. Cleaning up...')
    if 'cap' in globals():
        cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def stop_detection():
    global stop_signal
    stop_signal = True

def start_detection(config):
    global cap, stop_signal
    stop_signal = False
    
    # Initialize UDP socket for CoT messages
    cot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize the webcam
    if isinstance(config.camera_source, str) and config.camera_source.startswith('rtsp'):
        cap = cv2.VideoCapture(config.camera_source)
    else:
        cap = cv2.VideoCapture(int(config.camera_source))
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Only create window if show_video is True
    if config.show_video:
        cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection', 640, 480)
        
    try:
        print(f"Started. Press 'q' to quit, 'r' to reset detections")
        if config.reset_timer > 0:
            print(f"Auto-resetting detections every {config.reset_timer} seconds")
        
        while not stop_signal:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if it's time to reset detections
            current_time = time.time()
            if config.reset_timer > 0 and (current_time - config.last_reset) >= config.reset_timer:
                config.detected_objects.clear()
                config.last_reset = current_time
                print(f"Auto-reset detections (timer: {config.reset_timer}s)")
            
            results = config.model(frame)
            
            # Process the results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf)
                    cls = int(box.cls)
                    class_name = result.names[cls]
                    
                    if (class_name in config.selected_objects and 
                        conf > config.confidence_threshold and 
                        class_name not in config.detected_objects):
                        cot_message = create_cot_message(
                            class_name, 
                            conf,
                            config.latitude,
                            config.longitude,
                            config.callsign
                        )
                        try:
                            cot_socket.sendto(cot_message.encode(), (config.cot_host, config.cot_port))
                            config.detected_objects.add(class_name)
                            print(f"Sent CoT to {config.cot_host}:{config.cot_port}")  # Debug print
                        except Exception as e:
                            print(f"Send error: {e}")
                    
                    # Only draw if show_video is True
                    if config.show_video:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{class_name} {conf:.2f}'
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            if config.show_video:
                cv2.imshow('Object Detection', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    config.detected_objects.clear()
            
            time.sleep(0.01)
    
    finally:
        if cap:
            cap.release()
        if config.show_video:
            cv2.destroyAllWindows()
        cot_socket.close()

def test_cot_connection(host, port):
    """Test if we can send a CoT message"""
    try:
        test_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        test_message = "<?xml version='1.0' encoding='UTF-8'?><event version='2.0' type='a-u-G'><point lat='0' lon='0'/></event>"
        test_socket.sendto(test_message.encode(), (host, port))
        print(f"Test message sent successfully to {host}:{port}")
        test_socket.close()
        return True
    except Exception as e:
        print(f"Test connection failed: {e}")
        return False

def main():
    global cap
    
    # Initialize configuration through UI
    config = DetectionConfig()
    config.root.mainloop()
    
    if not config.ready_to_start:
        return
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Test CoT connection before starting
    if not test_cot_connection(config.cot_host, config.cot_port):
        print("Warning: CoT connection test failed")
    
    # Start detection using the unified function
    start_detection(config)

if __name__ == "__main__":
    main() 