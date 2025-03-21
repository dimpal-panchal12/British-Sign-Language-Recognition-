#imports.py already includes the necessary libraries
from imports import train_test_split, to_categorical, np, os

# Actions array 
actions = np.array([
    'hello', 'thanks', 'how_are_you', 'sorry', 'eat', 'sleep', 'drink', 'help',
    'happy', 'toilet', 'database', 'danger', 'cake', 'lab', 'laptop', 'teacher',
    'nation', 'yellow', 'fish_and_chips', 'magnet', 'table', 'umbrella', 'garlic',
    'post_code', 'zebra', 'kettle', 'london', 'river', 'bus', 'angry'
])

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('BSL_Data')

no_sequences = 30
sequence_length = 30

# mapping actions to numerical labels
label_mapping = {action: index for index, action in enumerate(actions)}

#creating lists to store sequences and their corresponding labels
feature_sequences, target_labels = [], []

#for loop to load each action 
for action in actions:
    for sequence_index in range(no_sequences):
        frame_sequence = []
        for frame_index in range(sequence_length):
            #loading the keypoint
            keypoint_path = os.path.join(DATA_PATH, action, str(sequence_index), f"{frame_index}.npy")
            keypoints = np.load(keypoint_path)
            frame_sequence.append(keypoints)
        
        #adding label to sequences
        feature_sequences.append(frame_sequence)
        target_labels.append(label_mapping[action])
        (label_mapping[action])

#converting lists to arrays
X = np.array(feature_sequences)
y = to_categorical(target_labels).astype(int)

#spliting data into 80:20 training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#to verify the shape of data 
print(f"Feature Set Shape: {X.shape}")
print(f"Labels Shape: {y.shape}")
print(f"Training Feature Set Shape: {X_train.shape}")
print(f"Testing Feature Set Shape: {X_test.shape}")
