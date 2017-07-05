from config import *
import classifier_pool as classifiers
from sklearn import preprocessing
import pickle

def main():
    print("Loading Data.")
    data = pickle.load(open(FEATURE_SAVE_PATH, 'rb'))
    featureMatrixTraining = data["featureMatrixTraining"]
    featureMatrixValidation = data["featureMatrixValidation"]    
    targetsTraining = data["targetsTraining"]
    targetsValidation = data["targetsValidation"]

    test_data = pickle.load(open(TEST_FEATURE_SAVE_PATH, 'rb'))
    featureMatrixTesting = test_data["featureMatrixTesting"]

    # Data Normalization
    if NORMALIZE_DATA:
        print("Normalizing Data.")
        normalizer = preprocessing.StandardScaler()
        featureMatrixTraining = normalizer.fit_transform(featureMatrixTraining)
        featureMatrixValidation = normalizer.transform(featureMatrixValidation)
        featureMatrixTesting = normalizer.transform(featureMatrixTesting)

    print('Classification.')
    print(configSummary + ", " + str(featureMatrixTraining.shape))
    if ESTIMATOR_POOL['svm'] is True:
        logReport = classifiers.svm(
            featureMatrixTraining, targetsTraining, featureMatrixValidation, featureMatrixTesting, targetsValidation)
        logReport = logReport + "\n"
    if ESTIMATOR_POOL['random_forest'] is True:
        logReport = classifiers.random_forest(
            featureMatrixTraining, targetsTraining, featureMatrixValidation, featureMatrixTesting, targetsValidation)
        logReport = logReport + "\n"

    with open(LOG_FILE, 'a+') as f:
        f.write("\n")
        f.write(configSummary + ", " + str(featureMatrixTraining.shape))
        f.write("\n")
        f.write(logReport)

if __name__ == '__main__':
    main()
