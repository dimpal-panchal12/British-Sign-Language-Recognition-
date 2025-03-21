from landmark_detection import custom_landmarks, extract_keypoints, mediapipe_detection
from imports import mp, np, os, cv2

#path for saving data 
DATA_PATH = os.path.join('BSL_Data')

#list of actions to be recorded
actions = np.array([
    'hello', 'thanks', 'how_are_you', 'sorry', 'eat', 'sleep', 'drink', 'help',
    'happy', 'toilet', 'database', 'danger', 'cake', 'lab', 'laptop', 'teacher',
    'nation', 'yellow', 'fish_and_chips', 'magnet', 'table', 'umbrella', 'garlic',
    'post_code', 'zebra', 'kettle', 'london', 'river', 'bus', 'angry'
])

no_sequences = 30
sequence_length = 30

#initializing video capture
cap = cv2.VideoCapture(1) #index of whichever camera is available 

try:
    #initializing mediapipe holistic model
    holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    #looping through 30 actions
    for action in actions:
        #looping through 30 sequences
        for sequence in range(no_sequences):
            print(f"Now recording: {action} (Sequence {sequence + 1}/{no_sequences})")  # Console feedback
            
            #looping through 30 frames
            for frame_num in range(sequence_length):
                
                #reading a frame
                ret, frame = cap.read()

                if not ret:
                    print("Failed to capture frame, skipping...")
                    continue

                #detecting landmarks
                image, landmarks = mediapipe_detection(frame, holistic_model)

                #drawing points and connecting lines on landmarks
                custom_landmarks(image, landmarks)
                
                #displaying text on screen
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                    #displaying frame 
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15,50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                    #displaying frame
                    cv2.imshow('OpenCV Feed', image)
                
                #exploring the keypoints as a NumPy array
                keypoints = extract_keypoints(landmarks)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                #exiting the OpenCV window
                if cv2.waitKey(10) & 0xFF == ord('x'):
                    break
            print(f"Completed collection for action '{action}' (Video {sequence + 1}/{no_sequences})")

finally:
    #closing all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
