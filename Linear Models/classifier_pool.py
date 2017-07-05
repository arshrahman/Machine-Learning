from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import hamming_loss, make_scorer
from os import environ as sysenv
import numpy as np
from config import *
import csv

def svm(training_feature_matrix, training_targets, validation_feature_matrix, testing_feature_matrix, validation_targets=None):
    from sklearn.svm import SVC
    # Parameter grid
    param_dist = {
        'kernel': ['rbf'],
        'gamma': np.logspace(-4, 2, 3),
        'C': np.logspace(-4, 2, 5),
    }
    svm = SVC(probability=True)
    clf = RandomizedSearchCV(
        estimator=svm,
        param_distributions=param_dist,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r" %
                  (mean_score, scores.std() * 2, params))

    logReport = "SVM [%0.3f +/-%0.3f][%0.3f +/-%0.3f]" % (clf.cv_results_['mean_test_score'][clf.best_index_], clf.cv_results_['std_test_score'][clf.best_index_], clf.cv_results_['mean_train_score'][clf.best_index_], clf.cv_results_['std_train_score'][clf.best_index_]) + "\n"
    logReport = logReport + "SVM %s" % (clf.best_params_) + "\n"

    svm = SVC(kernel='rbf', C=clf.best_params_['C'], gamma=clf.best_params_['gamma'])
    svm.fit(training_feature_matrix, training_targets)
    predicted_labels = svm.predict(validation_feature_matrix)
    if validation_targets is not None: # Evaluate performance on the validation data.
        logReport = logReport + "SVM Validation Accuracy [%0.3f]" % ((predicted_labels == validation_targets).mean())

    test_results = svm.predict(testing_feature_matrix)
    ExportCSV(test_results)

    print(logReport)
    return logReport

def random_forest(
    training_feature_matrix,
    training_targets,
    validation_feature_matrix,
    testing_feature_matrix,
    validation_targets=None 
):
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import randint as sp_randint
    # Parameter distribution for random search.
    param_dist = {
        "max_depth": [20, 25, 30, 35],
        "max_features": sp_randint(
            1,
            min(300, training_feature_matrix.shape[1])
        ),
        "min_samples_split": sp_randint(2, 20),
        "min_samples_leaf": sp_randint(2, 20),
        'n_estimators': [50, 100, 200, 250, 300],
        'bootstrap': [True, False],
    }

    r_forest = RandomForestClassifier(random_state=1)
    n_iter_search = 15
    clf = RandomizedSearchCV(
        estimator=r_forest,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        n_jobs=N_JOBS,
        pre_dispatch=N_JOBS,
        cv=CV_N,
        random_state=1,
    )
    clf.fit(training_feature_matrix, training_targets)

    if PRINT_ESTIMATOR_RESULTS is True:
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean_score, scores.std() * 2, params))

    logReport = "RANDOM FOREST [%0.3f +/-%0.3f][%0.3f +/-%0.3f]" % (clf.cv_results_['mean_test_score'][clf.best_index_], clf.cv_results_['std_test_score'][clf.best_index_], clf.cv_results_['mean_train_score'][clf.best_index_], clf.cv_results_['std_train_score'][clf.best_index_]) + "\n"
    logReport = logReport + "RANDOM FOREST %s" % (clf.best_params_) + "\n"

    
    r_forest = RandomForestClassifier(bootstrap=clf.best_params_['bootstrap'], max_depth=clf.best_params_['max_depth'], max_features=clf.best_params_['max_features'], 
                            min_samples_leaf= clf.best_params_['min_samples_leaf'], min_samples_split= clf.best_params_['min_samples_split'],
                           n_estimators = clf.best_params_['n_estimators'])
    
    #r_forest = RandomForestClassifier(bootstrap=False, max_depth=25, max_features=23, min_samples_leaf= 3, min_samples_split= 2, n_estimators = 250)
    r_forest.fit(training_feature_matrix, training_targets)
    predicted_labels = r_forest.predict(validation_feature_matrix)
    if validation_targets is not None: # Evaluate performance on the validation data.
        logReport = logReport + "RANDOM FOREST Validation Accuracy [%0.3f]" % ((predicted_labels == validation_targets).mean())

    test_results = r_forest.predict(testing_feature_matrix)
    ExportCSV(test_results)

    print(logReport)
    return logReport


def ExportCSV(Results):
    index = 1;
    with open(OUTPUT_PATH, 'w') as csvfile:   
    #configure writer to write standard csv file
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id', 'Prediction'])
        for row in Results:
            writer.writerow([index, row])
            index+=1


