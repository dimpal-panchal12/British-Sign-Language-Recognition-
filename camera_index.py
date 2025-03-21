import cv2

#for checking the index where camera is available
for i in range(5):  
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        cap.release()
    else:
        print(f"No camera found at index {i}")
