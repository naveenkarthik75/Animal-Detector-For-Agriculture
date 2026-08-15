Animal & Human Detector

An AI-powered real-time detection system that identifies animals and humans using a webcam feed. This project uses YOLO (You Only Look Once) for object detection and provides instant visual output with optional alerts.

FEATURES:
Real-time detection using webcam
Detects animals and humans
Fast and accurate predictions using YOLO
Simple and interactive UI (Streamlit)
Optional sound alert for detection
TECH STACK:
Python
OpenCV
YOLO (Ultralytics)
Streamlit
NumPy
Pygame (for alerts)
PROJECT STRUCTURE:
animal-detector/
│── app.py              # Main application
│── model/              # YOLO model files
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
INSTALLATION:
Clone the repository:
git clone https://github.com/your-username/animal-detector.git
cd animal-detector
Create virtual environment:
python -m venv venv
Activate environment:
venv\Scripts\activate   # Windows
source venv/bin/activate  # Mac/Linux
Install dependencies:
pip install -r requirements.txt
USAGE:
Allow webcam access
Detection starts automatically
Detected objects are shown with bounding boxes
RUN THE APPLICATION:
  COMMAND-streamlit run app.py

WORKING:
The webcam captures live video frames
Frames are processed using the YOLO model
The model detects objects (animals/humans)
Bounding boxes and labels are displayed in real-time
FUTURE IMPROVEMENT:
Add more animal classes
Improve detection accuracy
Deploy as a web app
Add notification system
