# Drowsiness Detection System

This project aims to detect driver drowsiness and alert the driver to prevent accidents. The system uses computer vision techniques to monitor the driver's eyes and detect signs of drowsiness. The project includes two versions: one for normal eye detection and another that works even when the driver is wearing sunglasses.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Description

The Drowsiness Detection System is a real-time monitoring system that detects drowsiness in drivers. When the driver's eye aspect ratio (EAR) falls below a certain threshold, the system triggers an alert to wake the driver. The project includes:
- `Drowsiness_Detection.py`: Detects drowsiness by tracing the eye.
- `Drowsiness_Detection(Sunglasses).py`: Detects drowsiness even when the driver is wearing sunglasses.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.6+
- OpenCV
- dlib
- imutils
- pygame
- scipy

You can install these using the following command:
```bash
pip install opencv-python dlib imutils pygame scipy
```
## Usage

1. **Run the Drowsiness Detection program:**
   - For normal eye detection:
     ```bash
     python Drowsiness_Detection.py
     ```
   - For detection with sunglasses:
     ```bash
     python Drowsiness_Detection(Sunglasses).py
     ```

2. **Press `ctrl+c` on the terminal to quit the program.**
