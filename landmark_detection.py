from imports import cv2, mp, np

def mediapipe_detection(frame, detection_model):
    #converting BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #detecting landmarks from frame
    landmarks = detection_model.process(frame_rgb)
    #converting RGB back to BGR
    processed_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR) 
    #returning the processed frame and landmarks detected 
    return processed_frame, landmarks


#for drawing points and connecting lines on the video
drawing_tools = mp.solutions.drawing_utils

def custom_landmarks(frame, landmarks):
    if landmarks.face_landmarks:
        drawing_tools.draw_landmarks(
            frame, 
            landmarks.face_landmarks, 
            mp.solutions.holistic.FACEMESH_CONTOURS,
            drawing_tools.DrawingSpec(color=(100,150,30), thickness=1, circle_radius=2),
            drawing_tools.DrawingSpec(color=(100,200,130), thickness=1, circle_radius=1)
        )
    if landmarks.pose_landmarks:
        drawing_tools.draw_landmarks(
            frame, 
            landmarks.pose_landmarks, 
            mp.solutions.holistic.POSE_CONNECTIONS,
            drawing_tools.DrawingSpec(color=(90,30,20), thickness=3, circle_radius=5),
            drawing_tools.DrawingSpec(color=(90,60,160), thickness=3, circle_radius=3)
        )
    if landmarks.left_hand_landmarks:
        drawing_tools.draw_landmarks(
            frame, 
            landmarks.left_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS,
            drawing_tools.DrawingSpec(color=(150,30,100), thickness=3, circle_radius=5),
            drawing_tools.DrawingSpec(color=(150,60,230), thickness=3, circle_radius=3)
        )
    
    if landmarks.right_hand_landmarks:
        drawing_tools.draw_landmarks(
            frame, 
            landmarks.right_hand_landmarks, 
            mp.solutions.holistic.HAND_CONNECTIONS,
            drawing_tools.DrawingSpec(color=(0,0,139), thickness=3, circle_radius=5),
            drawing_tools.DrawingSpec(color=(0,0,255), thickness=3, circle_radius=3)
        )

def extract_keypoints(landmarks):
    #extracting keypoints and filling with 0 when not detected
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in landmarks.pose_landmarks.landmark]).flatten() if landmarks.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in landmarks.face_landmarks.landmark]).flatten() if landmarks.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in landmarks.left_hand_landmarks.landmark]).flatten() if landmarks.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in landmarks.right_hand_landmarks.landmark]).flatten() if landmarks.right_hand_landmarks else np.zeros(21 * 3)
    
    #concatinating all keypoints into a single array
    keypoints = np.concatenate([pose, face, lh, rh])
    #returning the keypoints
    return keypoints





