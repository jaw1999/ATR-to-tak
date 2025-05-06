# Object Detection Suite

A collection of Python scripts for object detection with CoT (Cursor on Target) integration.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Scripts

### 1. Simple Object Tracker (object_tracker.py)
Basic real-time object detection using your webcam.

```bash
python object_tracker.py
```
- Press 'q' to quit
- Press 'f' to toggle fullscreen

### 2. Web Interface Detector (scripts/object_detector_notify.py)
Object detection with web configuration and CoT notifications.

Start the web server:
```bash
python scripts/web_server.py
```

Then open `http://localhost:8080` in your browser to configure:
- Detection location (lat/long)
- CoT server details (IP/port)
- Objects to detect
- Confidence threshold
- Camera settings
- Detection reset timer


```

