from flask import Flask, render_template, request, jsonify
import threading
import object_detector_notify as detector
import json
import cv2
import time

app = Flask(__name__)

# Global variables to manage detector state
detector_thread = None
detector_config = None
is_running = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_detector():
    global detector_thread, detector_config, is_running
    
    if is_running:
        return jsonify({"status": "error", "message": "Detector already running"})
    
    # Get configuration from POST request
    config_data = request.json
    
    # Create detector configuration
    detector_config = detector.DetectionConfig()
    detector_config.latitude = float(config_data.get('latitude', 0.0))
    detector_config.longitude = float(config_data.get('longitude', 0.0))
    detector_config.cot_host = config_data.get('cot_host', '127.0.0.1')
    detector_config.cot_port = int(config_data.get('cot_port', 6969))
    detector_config.callsign = config_data.get('callsign', 'Detection1')
    detector_config.confidence_threshold = float(config_data.get('confidence', 0.5))
    detector_config.selected_objects = set(config_data.get('selected_objects', []))
    detector_config.camera_source = config_data.get('camera_source', 0)
    detector_config.show_video = config_data.get('show_video', True)
    detector_config.reset_timer = int(config_data.get('reset_timer', 0))
    detector_config.last_reset = time.time()
    
    # Start detector in a separate thread
    detector_thread = threading.Thread(target=detector.start_detection, args=(detector_config,))
    detector_thread.start()
    is_running = True
    
    return jsonify({"status": "success", "message": "Detector started"})

@app.route('/stop', methods=['POST'])
def stop_detector():
    global detector_thread, is_running
    
    if not is_running:
        return jsonify({"status": "error", "message": "Detector not running"})
    
    detector.stop_detection()
    detector_thread.join()
    is_running = False
    
    return jsonify({"status": "success", "message": "Detector stopped"})

@app.route('/status')
def get_status():
    if detector_config:
        config_dict = {
            'latitude': detector_config.latitude,
            'longitude': detector_config.longitude,
            'cot_host': detector_config.cot_host,
            'cot_port': detector_config.cot_port,
            'callsign': detector_config.callsign,
            'confidence_threshold': detector_config.confidence_threshold,
            'selected_objects': list(detector_config.selected_objects),
            'camera_source': detector_config.camera_source,
            'show_video': detector_config.show_video
        }
    else:
        config_dict = None
        
    return jsonify({
        "running": is_running,
        "config": config_dict
    })

@app.route('/objects')
def get_objects():
    # Create temporary model to get object names
    model = detector.YOLO('yolov8n.pt', verbose=False)
    return jsonify(list(model.names.values()))

@app.route('/cameras')
def get_cameras():
    cameras = get_available_cameras()
    return jsonify(cameras)

def get_available_cameras():
    """Query available camera devices"""
    available_cameras = []
    # Only try first few cameras to avoid spam
    for i in range(2):  # Just check cameras 0 and 1
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append({
                        'id': i,
                        'name': f'Camera {i}'
                    })
                cap.release()
        except:
            pass
    return available_cameras

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080) 