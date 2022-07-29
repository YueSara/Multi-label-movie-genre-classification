import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.ensemble import RakelD
from sklearn.svm import LinearSVC
from sklearn import metrics
from utils import custom_train_test_split


#### CLASSIFIERS ####
def classifierchain(X_train_tfidf, y_train, X_test_tfidf):
    clf = ClassifierChain(
	    classifier=LinearSVC(C=1, loss='hinge', random_state=0),
	    require_dense=[False, True]
	)
    print('Fitting Classifier Chain')
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    return clf, y_pred


def rakeld(X_train_tfidf, y_train, X_test_tfidf):
    clf = RakelD(
        labelset_size=2,
        base_classifier=LinearSVC(C=1, loss='squared_hinge', random_state=0),
        base_classifier_require_dense=[False, True]
    )
    print('Fitting RAkEL D')
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    return clf, y_pred


#### TRAINING ####
def train_baseline(dataset_name):
    #### DATA ####
    data = pd.read_pickle('./data/'+dataset_name+'.pkl')

    # binarize labels
    mlb = MultiLabelBinarizer()
    y_binarized = mlb.fit_transform(data['genres'])

    # split
    X_train, y_train, X_test, y_test = custom_train_test_split(data['plots'], y_binarized, test_size=0.2, order=1)
    print('Data split done')

    # tf-idf
    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    # for later inference, already saved in models directory
    #joblib.dump(tfidf, './models/tfidf_'+dataset_name+'.pkl')
    #print('Saved tfidf object')

    # fit classifier
    if DATASET_NAME=='kaggle':
        clf, y_pred = classifierchain(X_train_tfidf, y_train, X_test_tfidf)
    else:
        clf, y_pred = rakeld(X_train_tfidf, y_train, X_test_tfidf)
    # for later inference, already saved in models directory
    #joblib.dump(clf, './models/baseclf_'+dataset_name+'.pkl')
    #print('Saved fitted classifier object')

    # get metrics and print results
    y_pred.toarray()
    report = metrics.classification_report(y_test, y_pred, output_dict=True, target_names=mlb.classes_)
    report = pd.DataFrame(report).T
    mets = {'f1': round(report.loc['samples avg', 'f1-score'], 6),
            'p': round(report.loc['samples avg', 'precision'], 6),
            'r': round(report.loc['samples avg', 'recall'], 6),
            'jacc': round(metrics.jaccard_score(y_test, y_pred, average='samples'), 6),
            'acc': round(metrics.accuracy_score(y_test, y_pred), 6),
            'ham': round(metrics.hamming_loss(y_test, y_pred), 6)}
    if DATASET_NAME == 'kaggle':
        print('Results for Kaggle Classifier Chain:')
    else:
        print('Results for CMU RAkEL D:')
    for k, v in mets.items():
        print(f'{k}: {v}')


if __name__ == "__main__":
    # UNCOMMENT CHOSEN DATASET
    DATASET_NAME = 'kaggle'
    #DATASET_NAME = 'cmu'
    train_baseline(DATASET_NAME)
