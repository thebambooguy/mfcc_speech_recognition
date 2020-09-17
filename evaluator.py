from FeatureExtractor import simple_mfcc, getting_commands_from_signals
from joblib import load

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PATH = 'dataset/'
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
	
def evaluator():

    dbreader = load("dbreader.mdl")
    test_commands = getting_commands_from_signals(dbreader.testing_signals,dbreader.testing_text)
    test_mfcc_data = simple_mfcc(test_commands)

    y_test = test_mfcc_data['command']
    X_test = test_mfcc_data.drop(columns=['command'])

    load_model = load("RandomForestModelparameters.joblib")

    predictions = load_model.predict(X_test)

    print(classification_report(y_test,predictions))
    plot_confusion_matrix(y_test,predictions)

if __name__ == "__main__":
    evaluator()
