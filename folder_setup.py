from imports import np, os

#path for collected data
DATA_PATH = os.path.join('BSL_Data')  

#list of selected 30 actions
actions = np.array([
    'hello', 'thanks', 'how_are_you', 'sorry', 'eat', 'sleep', 'drink', 'help',
    'happy', 'toilet', 'database', 'danger', 'cake', 'lab', 'laptop', 'teacher',
    'nation', 'yellow', 'fish_and_chips', 'magnet', 'table', 'umbrella', 'garlic',
    'post_code', 'zebra', 'kettle', 'london', 'river', 'bus', 'angry'
])

#number of sequences to be recorded for each action
no_sequences = 30

#number of frames in each sequence
sequence_length = 30

#creating folders and sub-folders for each action 
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass
