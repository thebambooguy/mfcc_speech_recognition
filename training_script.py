from DbReader import DbReader
from FeatureExtractor import simple_mfcc, getting_commands_from_signals
from Model import RandomForestModel
import joblib

PATH = 'dataset/283335'
ONE_PERSON_SPLIT = 0.5
WHOLE_DATASET_SPLIT = 0.3

def train_script():
    dbreader = DbReader(PATH, split_size = ONE_PERSON_SPLIT)
    commands = getting_commands_from_signals(dbreader.training_signals, dbreader.training_text)
    mfcc_data = simple_mfcc(commands)

    y_train = mfcc_data['command']
    X_train = mfcc_data.drop(columns=['command'])

    rf_model = RandomForestModel()

    model_to_fit = rf_model.gridsearchCV()
    model_to_fit.fit(X_train, y_train)
    rf_model.set_internal_model(model_to_fit.best_estimator_)
    print(model_to_fit.best_estimator_)
    rf_model.save_model()
 
    joblib.dump(dbreader, "dbreader.mdl")

if __name__ == "__main__":
    train_script()
