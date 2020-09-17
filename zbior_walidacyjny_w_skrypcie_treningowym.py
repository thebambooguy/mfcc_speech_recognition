from DbReader import DbReader
from FeatureExtractor_2 import simple_mfcc, getting_commands_from_signals
from Model import LogisticRegressionModel, RandomForestModel
 
import joblib

import matplotlib.pyplot as plt
from sklearn.metrics import  confusion_matrix
import numpy as np
 
PATH = '../dataset/283335'
ONE_PERSON_SPLIT = 0.5
WHOLE_DATASET_SPLIT = 0.3
 
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test,y_pred)
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j) / float(a))
        norm_conf.append(tmp_arr)
 
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
 
    width, height = cm.shape
 
    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
 
    cb = fig.colorbar(res)
    commands = np.unique(y_test)
    plt.xticks(range(width), range(width))
    plt.yticks(range(height), commands[:width])
    plt.savefig('confusion_matrix.png', format='png')
 
def train_script_2():
     
    dbreader = DbReader(PATH, split_size = ONE_PERSON_SPLIT)
    training_commands = getting_commands_from_signals(dbreader.training_signals[:2],dbreader.training_text[:2])
    valid_commands = getting_commands_from_signals(dbreader.training_signals[2:], dbreader.training_text[2:])
 
    training_mfcc_data = simple_mfcc(training_commands)
    valid_mfcc_data = simple_mfcc(valid_commands)
 
    y_train = training_mfcc_data['command']
    X_train = training_mfcc_data.drop(columns=['command'])
 
    y_valid = valid_mfcc_data['command']
    X_valid = valid_mfcc_data.drop(columns=['command'])
 
 
    rf_model = RandomForestModel()
 
    model_to_fit = rf_model.gridsearchCV()
    model_to_fit.fit(X_train, y_train)
    rf_model.set_internal_model(model_to_fit.best_estimator_)
    print(model_to_fit.best_estimator_)
    rf_model.save_model()
 
    joblib.dump(dbreader, "dbreader.mdl")
 
     
    predictions = rf_model.predict(X_valid)
    plot_confusion_matrix(y_valid, predictions)
 
if __name__ == "__main__":
    train_script_2()
