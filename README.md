# Anti-Spoofing Face Detection System using Intel RealSense Camera

This project utilizes Intel RealSense camera and depth sensing to perform anti-spoofing face detection.

## Overview

This Python-based system integrates the RealSense camera to detect faces and determine whether a real human face is present in the frame. It uses Dlib for face detection and facial landmark identification. The depth information obtained from the RealSense camera helps in distinguishing real human faces from spoof attempts.

## Requirements

- Python 3.x
- Dlib
- OpenCV
- PyRealSense2
- Intel RealSense Camera

## Installation

1. Install Python (if not already installed): [Python Installation Guide](https://www.python.org/downloads/)
2. Install required Python libraries:

    ```bash
    pip install dlib opencv-python imutils pyrealsense2
    ```
3. Connect and configure the Intel RealSense camera.

## Usage

1. Clone this repository:

    ```bash
    git clone https://github.com/HasarinduPerera/realsense-antispoofing-facedetection.git
    ```

2. Run the Python script:

    ```bash
    python main.py
    ```

3. Ensure the RealSense camera is properly positioned to capture faces.

4. View the live feed from the camera showing face detections and anti-spoofing checks.

## How It Works

1. Camera Initialization and Configuration
The code initializes the Intel RealSense camera using the pyrealsense2 library. It configures the camera to capture both color and depth streams at a resolution of 640x480 and a frame rate of 30 frames per second.

2. Face Detection and Landmark Identification
Using the dlib library, the system performs face detection on the captured color frames. Detected faces are then processed to identify facial landmarks, such as eyes, nose, and ears, which are crucial for anti-spoofing checks.

3. Depth Analysis for Anti-Spoofing
Depth information obtained from the RealSense camera is used to differentiate between real human faces and spoof attempts. The depth data is associated with specific facial landmarks to calculate distances, particularly between the nose and ears. These distances are crucial in determining whether a face is physically present or is a static image or mask.

4. Anti-Spoofing Validation
The system establishes threshold values for the calculated distances between facial landmarks. If the calculated values fall within the defined thresholds, it confirms the presence of a real human face. Otherwise, it raises an alert, indicating a potential spoof attempt.

## Important Notes

- Ensure proper lighting conditions for accurate face detection.
- Adjust the camera position and angle for optimal results.
- Depth information is crucial for anti-spoofing checks.

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork it and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).