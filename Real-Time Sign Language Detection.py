from preprocessing import actions
from tensorflow.keras.models import load_model
from imports import cv2, np, mp, time
from landmark_detection import  mediapipe_detection, custom_landmarks, extract_keypoints

#loading trained model
model = load_model('final_model_gru_best_params.keras') 

#initializing video capture
capture = cv2.VideoCapture(1) #index of whichever camera is available

if not capture.isOpened():
    print("Failed to open camera. Exiting...")
    exit()

#wait time of 2 seconds
time.sleep(2)

#initializing holistic model
holistic_model = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

sequence = []
sequence_length = 30  
threshold = 0.6  #confidence threshold

try:
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break

        image, landmarks = mediapipe_detection(frame, holistic_model)
        custom_landmarks(image, landmarks)
        keypoints = extract_keypoints(landmarks)
        sequence.append(keypoints)

        sequence = sequence[-sequence_length:]

        if len(sequence) == sequence_length:
            #pridicting the gesture
            sequence_input = np.expand_dims(sequence, axis=0)
            predictions = model.predict(sequence_input)
            predicted_class = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class]

            #checking confidence
            if confidence > threshold:
                cv2.putText(image, f"{actions[predicted_class]} ({confidence:.2f})", (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('British Sign Language Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('x'):
            break

finally:
    if holistic_model is not None:
        holistic_model.close()
    capture.release()
    cv2.destroyAllWindows()
