import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import librosa
import scipy.io.wavfile as wav


PATH = 'dataset/'
ONE_PERSON_SPLIT = 0.5
WHOLE_DATASET_SPLIT = 0.3


class DbReader:
    def __init__(self, files_dir=None, split_size=None):
        self._files_dir = files_dir
        self._wav_signals, self._rate = self.importing_wav_files(num_of_files_per_person=8, assert_standard_srate=True,
                                                                 assert_convert_to_mono=True)

        self._text_data = self.importing_text_files(num_of_files_per_person=8)
        self._training_signals, self._testing_signals, self._training_text, self._testing_text = self.splitting_files(self._wav_signals, self._text_data, split_size=split_size)


    @property
    def training_signals(self):
        return self._training_signals

    @property
    def testing_signals(self):
        return self._testing_signals

    @property
    def training_text(self):
        return self._training_text

    @property
    def testing_text(self):
        return self._testing_text

    @property
    def rate(self):
        return self._rate


    def importing_wav_files(self, num_of_files_per_person, assert_standard_srate=True, assert_convert_to_mono=True):

        signals = []
        signal_for_same_person = []
        for root, dirs, files in os.walk(self._files_dir):
            for single_file in files:
                if single_file.endswith('.wav'):
                    (sig, rate) = librosa.load(root + '/' + single_file, sr=44100)
                    if assert_convert_to_mono:
                        if sig.ndim == 2:
                            sig = np.mean(sig, axis=1)
                    # Normalization
                    sig = sig - np.mean(sig)
                    sig = sig / np.max(sig)
                    signal_for_same_person.append(sig)
                if len(signal_for_same_person) == num_of_files_per_person:
                    signals.extend(signal_for_same_person)
                    signal_for_same_person = []

        signals_as_array = np.asarray(signals)

        return signals_as_array, rate

    def importing_text_files(self, num_of_files_per_person):

        text_for_same_person = []
        text_data = []

        for root, dirs, files in os.walk(self._files_dir, topdown=False):
            for single_file in files:
                   if single_file.lower().endswith('.txt'):
                       text = pd.read_csv(root + '/' + single_file, sep='\t', names=['Start', 'End', 'Command'], header=None).values
                       text_for_same_person.append(text)
                       if len(text_for_same_person) == num_of_files_per_person:
                           text_data.extend(text_for_same_person)
                           text_for_same_person = []

        text_data = np.asarray(text_data)

        return text_data

    def splitting_files(self, signals_as_array, text_data, split_size):

        training_signals, testing_signals, training_texts, testing_texts = train_test_split(signals_as_array, text_data,
                                                                                            test_size=split_size,
                                                                                            random_state=42)
#        print(training_signals.shape,testing_signals.shape)

#        testing_signals = testing_signals.transpose()
        training_dfs = self.creating_dataframes_for_text_files(training_texts)
        testing_dfs = self.creating_dataframes_for_text_files(testing_texts)

        return training_signals, testing_signals, training_dfs, testing_dfs

    def creating_dataframes_for_text_files(self, text_data):

        text_dfs = []
        for i, info_file in enumerate(text_data):
            temp = pd.DataFrame(info_file, columns=['Start', 'End', 'Command'])
            temp['Start'] = temp['Start'].astype(float)
            temp['End'] = temp['End'].astype(float)
            text_dfs.append(temp)

        return text_dfs


if __name__ == "__main__":
    dbreader = DbReader(PATH, ONE_PERSON_SPLIT)
    signals, rate = dbreader.importing_wav_files(8)
    text_data = dbreader.importing_text_files(8)
    print(signals)
#    print(len(text_data))
    print(dbreader._wav_signals.shape)
    print(dbreader.training_signals.shape)
#    print(len(dbreader.training_text))
#    print(dbreader.training_text[0].head())
