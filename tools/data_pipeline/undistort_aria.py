import cv2
import numpy as np
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Undistort video using camera calibration parameters')
parser.add_argument('--input', type=str, required=True, help='Path to input video file')
parser.add_argument('--output', type=str, required=True, help='Path to output video file')
args = parser.parse_args()

# Camera matrix and distortion coefficients
camera_matrix = np.array([
    [575.3553126527711, 0.0, 705.9916372212775],
    [0.0, 582.8534148749328, 705.9151539796964],
    [0.0, 0.0, 1.0]
])

dist_coeffs = np.array([[-0.03467338458503362, -0.12326126173118053, 
                         0.005485937530667466, 0.0009531354921084886, 
                         0.06385932503685757]])

# Read input video
cap = cv2.VideoCapture(args.input)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, 
                                                      (frame_width, frame_height), 
                                                      alpha=1, 
                                                      newImgSize=(frame_width, frame_height))

# Set up video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    x, y, w, h = roi
    undistorted_frame = undistorted_frame[y:y+h, x:x+w]  
    undistorted_frame = cv2.resize(undistorted_frame, (frame_width, frame_height))
    out.write(undistorted_frame)

cap.release()
out.release()
print("Undistorted video saved to:", args.output)