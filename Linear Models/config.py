from os import environ as sysenv

ESTIMATOR_POOL = {
    'svm': False,
    'random_forest': True,
}
APPLY_MASK = True
FEATURE_POOL = {
    'hog': False,
    'sift': False,
    'sift_dense': True,
    'skeleton': False,
}
DATA_MODALITY = 'depth'

FEATURE_SAVE_PATH = "./features/sift_dense_px9.pkl"
TEST_FEATURE_SAVE_PATH = "./features/test_sift_dense_px9.pkl"

TRAINING_DATA = "./data/a1_dataTrain.pkl"
#TRAINING_DATA = "./data/data-1-150-10-cut.pkl"
VALIDATION_DATA = None
#VALIDATION_DATA = "./data/data-301-320-10-cut.pkl"

#TRAINING_DATA = "./data/data-301-320-10-cut.pkl"
#VALIDATION_DATA = None
#VALIDATION_DATA = "./data/data-demo.pkl"

TEST_DATA = "./data/a1_dataTest.pkl"
OUTPUT_PATH = "./output/sift_dense__px9_rfc_results.csv"

# If VALIDATION_DATA is None, then a hold out set is created from the training data.
VALIDATION_DATA_RATIO = 0.2

NORMALIZE_DATA = True

# Num of CPU cores for parallel processing.
N_JOBS = 16
if 'LBD_N_JOBS' in sysenv.keys():
    N_JOBS = int(sysenv['LBD_N_JOBS'])
    print('Using %d jobs' % N_JOBS)
# If True, prints results for all possible configurations.
PRINT_ESTIMATOR_RESULTS = True
# How many cross validation folds to do
CV_N = 5

LOG_FILE = "./classification_results.txt"

# configSummary of the configuration
"""
"Train data, validation data, mask, modality, feature extractor"
configSummaryFormat = "[Train:%s][Validation:%s][%s][%s][%s]"
configSummaryParameters = [TRAINING_DATA.split("/")[-1]]
if VALIDATION_DATA is None:
    configSummaryParameters.append(TRAINING_DATA.split("/")[-1])
else:
    configSummaryParameters.append(VALIDATION_DATA.split("/")[-1])
if APPLY_MASK:
    configSummaryParameters.append("Masked")
else:
    configSummaryParameters.append("Not Masked")
"""

configSummary = "Train: " + TRAINING_DATA.split("/")[-1]
if VALIDATION_DATA is None:
    configSummary = configSummary + ", Validation: Same, "
else:
    configSummary = configSummary + ", Validation: " + VALIDATION_DATA.split("/")[-1] + ", "

if FEATURE_POOL['skeleton'] is False and APPLY_MASK is True:
    configSummary = configSummary + "masked "
configSummary = configSummary + DATA_MODALITY + ' data'
for key in FEATURE_POOL.keys():
    if FEATURE_POOL[key]:
        configSummary = configSummary  + ", " + key
configSummary = configSummary + " features"

if NORMALIZE_DATA:
    configSummary = ", " + configSummary + " normalized"
