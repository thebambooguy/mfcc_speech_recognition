from DbReader import DbReader
import numpy as np
import pandas as pd
from python_speech_features import mfcc
from tqdm import tqdm


PATH = 'dataset/'
ONE_PERSON_SPLIT = 0.5
WHOLE_DATASET_SPLIT = 0.3
RATE = 44100
SIZE_OF_WINDOW_MFCC = 0.01
 
 
def getting_commands_from_signals(signals, text_data):
    commands = []
    for num_signal, signal in enumerate(signals):
        for i in range(len(text_data[num_signal])):
            command_start = int(np.round(RATE * text_data[num_signal].Start[i]))
            command_end = int(np.round(RATE * text_data[num_signal].End[i]))
            command = signal[command_start:command_end]
            commands.append((command,text_data[num_signal].Command[i]))
 
    return commands
 
def padding_command(command, size_to_pad=65000):
    if len(command) < 65000:
        command = np.pad(command, (0, size_to_pad - len(command)), 'linear_ramp')
    else:
        command = command[:size_to_pad]
    return command

def reshape_mfcc_frame(mfcc_features):
    rows, columns = mfcc_features.shape
    reshaped_mfcc = pd.DataFrame(mfcc_features.reshape(1, rows* columns))
    return reshaped_mfcc

def simple_mfcc(commands):
    mfcc_data = pd.DataFrame()
    for command in commands:
        padded_command = padding_command(command[0], size_to_pad=65000)
        mfcc_features = mfcc(padded_command, RATE, winlen=0.01)
        mfcc_feat_df = reshape_mfcc_frame(mfcc_features)
        mfcc_feat_df['command'] = command[1]
        mfcc_data = mfcc_data.append(mfcc_feat_df)
    mfcc_data = mfcc_data.reset_index(drop=True)
    return mfcc_data
 
if __name__ == "__main__":
    dbreader = DbReader(PATH, WHOLE_PERSON_SPLIT)
    commands = getting_commands_from_signals(dbreader.training_signals,dbreader.training_text)
    print(len(commands))
    print(commands[0])
    mfcc_data = simple_mfcc(commands)
    print(mfcc_data.head)
