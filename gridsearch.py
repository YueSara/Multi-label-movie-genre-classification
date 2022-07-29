import pandas as pd
import numpy as np
from time import time
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import IterativeStratification
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.ensemble import RakelD
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from utils import custom_train_test_split


if __name__ == "__main__":
    # UNCOMMENT CHOSEN DATASET
    DATASET_NAME = 'kaggle'
    #DATASET_NAME = 'cmu'

    # where to save results of the grid search
    RESULTS_PATH = './grid_results.pkl'

    #### DATA ####
    data = pd.read_pickle('./data/'+DATASET_NAME+'.pkl')

    # binarize labels
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(data['genres'])

    X_train, y_train, X_test, y_test = custom_train_test_split(data['plots'], y_binarized, test_size=0.2, order=1)
    print('Data split done')


    #### DEFINE GRID ####
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ('clf', None)
    ])

    # grid for Classifier Chain and Label Powerset
    param_grid = [{
        #'tfidf__max_features': [20000, 10000],     # additionally: [33000, 45000] for CMU
        #'tfidf__ngram_range': [(1, 2), (1, 3)],
        'clf': [ClassifierChain(require_dense=[False,True]), LabelPowerset(require_dense=[False,True])],
        'clf__classifier': [LinearSVC(random_state=0)],
        'clf__classifier__loss': ['hinge', 'squared_hinge'],
        #'clf__classifier__class_weight': [None, 'balanced'],
        #'clf__classifier__penalty': ['l1', 'l2'],
        'clf__classifier__C': [0.01, 0.1, 1, 10, 100]
    }]

    # grid for RAkEL D
    #param_grid = [{
    #    #'tfidf__max_features': [20000, 10000],
    #    #'tfidf__ngram_range': [(1, 2), (1, 3)],
    #    'clf': [RakelD()],
    #    'clf__labelset_size': [2, 3, 4, 10, 15],    # [2, 3, 10, 20, 30] for CMU
    #    'clf__base_classifier': [LinearSVC(random_state=0)],
    #    'clf__base_classifier__loss': ['hinge', 'squared_hinge'],
    #    'clf__base_classifier_require_dense': [[False, True]],
    #    # 'clf__classifier__penalty': ['l1', 'l2'],
    #    # 'clf__classifier__dual': [True],
    #    'clf__base_classifier__C': [0.01, 0.1, 1, 10, 100]
    #}]

    cv = IterativeStratification(n_splits=3)

    hamming = make_scorer(metrics.hamming_loss, greater_is_better=False)
    scoring = {'hamming': hamming, 'acc': 'accuracy', 'f1s': 'f1_samples', 'jaccs': 'jaccard_samples'}

    grid_search = GridSearchCV(pipe, param_grid=param_grid, scoring=scoring, n_jobs=-2, verbose=10, cv=cv, refit='f1s',
                               error_score=0)
    print('Grid has been defined, now starting grid search...')

    #### FITTING ####
    start = time()
    grid_search.fit(X_train, y_train)
    end = time()

    # save results
    joblib.dump(grid_search.cv_results_, RESULTS_PATH)
    print('Results saved')

    print(f'GridSearch took {end - start} seconds.\n')
    print(f'Best score: {grid_search.best_score_}\n')
    best_parameters = grid_search.best_estimator_.get_params()
    print('Best parameters:')
    for param_name in sorted(param_grid[0].keys()):
        print(f"\t{param_name}: \t{best_parameters[param_name]}")