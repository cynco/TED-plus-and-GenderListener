# These functions are from Emmanuel Contreras-Campana, Ph.D.

# Import common python libraries
import os

# Import numpy library
import numpy as np

# Import scipy library
import scipy.sparse as sp

# Import matplotlib library
import matplotlib as mpl

mpl.use('TkAgg') # Change MacOSX backend

import matplotlib.pyplot as plt

# Import panda library
import pandas as pd

# Import scikit-learn
import sklearn

from sklearn.externals import joblib

from sklearn.model_selection import GridSearchCV, learning_curve

from sklearn.metrics import (log_loss,  precision_score,
                             recall_score, accuracy_score,
                             precision_recall_fscore_support)

# Import jupyter library
from IPython.core.display import display


# Define summary report
def summary_report(estimator, X_test, X_train, y_test, y_train):
    """
    Summary report listing accuracy, precision, and log loss

    Parameters
    ----------
    estimator : array, shape = [n_samples]
    true class, intergers in [0, n_classes - 1)
    X_test : array,  shape = [n_samples, n_classes]
    X_train : array,  shape = [n_samples, n_classes]
    y_train : array,  shape = [n_samples, n_classes]
    y_test : array,  shape = [n_samples, n_classes]

    Returns
    -------
    None : returns None
    """

    # Calculate predictions on training and test data
    y_train_predict = estimator.predict(X_train)
    y_train_predict_proba = estimator.predict_proba(X_train)

    y_test_predict = estimator.predict(X_test)
    y_test_predict_proba = estimator.predict_proba(X_test)

    # Print summary report of scores
    accuracy_train = accuracy_score(y_train, y_train_predict)
    print('Accuracy score on train data:', accuracy_train)

    accuracy = accuracy_score(y_test, y_test_predict)
    print('Accuracy score on test data:', accuracy, '\n')

    precision_train = precision_score(y_train, y_train_predict, average=None)
    print('Precision score on train data:', precision_train)

    precision = precision_score(y_test, y_test_predict, average=None)
    print('Precision score on test data:', precision, '\n')

    recall_train = recall_score(y_train, y_train_predict, average=None)
    print('Recall score on train data:', recall_train)

    recall = recall_score(y_test, y_test_predict, average=None)
    print('Recall score on test data:', recall, '\n')

    logloss_train = log_loss(y_train, y_train_predict_proba)
    print('Log loss on train data:', logloss_train)

    logloss = log_loss(y_test, y_test_predict_proba)
    print('Log loss on test data:', logloss)

    scoresDict = {'accuracy_train':accuracy_train, 'precision_train':precision_train, 'recall_train':recall_train, 'logloss_train':logloss_train, 'accuracy':accuracy, 'precision':precision, 'recall':recall, 'logloss':logloss}

    return scoresDict

# Extract estimator name
def extract_estimator_name(estimator):

    # check to see if estimator is a pipeline object or not
    if isinstance(estimator, sklearn.pipeline.Pipeline):
        data_type = type(estimator._final_estimator)

    # check to see if estimator is a grid search object or not
    elif isinstance(estimator, sklearn.model_selection._search.GridSearchCV):
        # check to see if estimator is a pipeline object or not
        if isinstance(estimator.best_estimator_, sklearn.pipeline.Pipeline):
            data_type = type(estimator.best_estimator_._final_estimator)

        else:
            data_type = type(estimator.best_estimator_)

    # object is not a pipeline or grid search estimator
    else:
        data_type = type(estimator)

    name = ''.join(filter(str.isalnum, str(data_type).split('.')[-1]))

    return name


# Standard nested k-fold cross validation
def grid_search(estimator, X, y, outer_cv, inner_cv,
                param_grid, scoring='accuracy',
                n_jobs=1, debug=False):
    """
    Nested k-fold cross-validation

    Parameters
    ----------
    estimator : array, shape = [n_samples]
         true class, integers in [0, n_classes - 1)
    X : array,   shape = [n_samples, n_classes]
    y : array,   shape = [n_samples, n_classes]
    outer_cv :   shape = [n_samples, n_classes]
    inner_cv :   shape = [n_samples, n_classes]
    param_grid : shape = [n_samples, n_classes]
    scoring :    shape = [n_samples, n_classes]
    n_jobs : int, default 1
    debug : boolean, default Fasle

    Returns
    -------
    grid : GridSearchCV object
        A post-fit (re-fitted to full dataset) GridSearchCV object where the estimator is a Pipeline.
    """

    outer_scores = []

    num_classes = len(list(set(y)))
    zeroes = [0]*num_classes
    pre_rec_f1_sup = (np.array(zeroes, dtype='float64'), np.array(zeroes, dtype='float64'),
                      np.array(zeroes, dtype='float64'), np.array(zeroes, dtype='int64'))

    # Extract model name
    name = extract_estimator_name(estimator).lower()

    # Set up grid search configuration
    grid = GridSearchCV(estimator=estimator, param_grid=param_grid,
                        cv=inner_cv, scoring=scoring, n_jobs=n_jobs)

    # Set aside a hold-out test data for model evaluation
    n = 0
    for k, (training_samples, test_samples) in enumerate(outer_cv.split(X, y)):

        # x training and test data
        if isinstance(X, pd.DataFrame):
            x_train = X.iloc[training_samples]
            x_test = X.iloc[test_samples]

        # in case of spare matrices
        else:
            x_train = X[training_samples]
            x_test = X[test_samples]

        # y training and test data
        if isinstance(y, pd.Series):
            y_train = y.iloc[training_samples]
            y_test = y.iloc[test_samples]

        # in case of numpy arrays
        else:
            y_train = y[training_samples]
            y_test = y[test_samples]

        # Build classifier on best parameters using outer training set
        # Fit model to entire training data (i.e tuning & validation dataset)
        print('Fold-', k+1, 'model fitting...')

        # Train on the training set
        grid.fit(x_train, y_train)

        # Hyper-parameters of the best model
        if debug:
            print('\n\t', grid.best_estimator_.get_params()[name])

        # Evaluate
        score = grid.score(x_test, y_test)

        outer_scores.append(abs(score))
        print('\n\tModel validation score:', abs(score), '\n')

        # Add the precision, recall, f1 score, and support of each outer fold
        # i.e. running total
        all_scores = precision_recall_fscore_support(y_test, grid.predict(x_test))

        pre_rec_f1_sup = [x + y for x, y in zip(pre_rec_f1_sup, all_scores)]

        print('\t', all_scores, '\n')

        n += 1

    # Print hyper-parameters of best model
    print('\n*** Hyper-paramters of best model:\n\n',
          grid.best_estimator_.get_params()[name])

    # Print final model evaluation (i.e. mean cross-validation scores)
    print('\n*** Final model evaluation (mean cross-val scores):',
          np.array(outer_scores).mean())

    # Print cross validated precision, recall, f1 score, and support
    performance = [s/n for s in pre_rec_f1_sup]
    performanceLabel = ['precision', 'recall', 'f1 score', 'support']
    print('\n Final precision, recall, f1 score, and support values:\n')
    for ii in range(0,len(performance)):
        print(performanceLabel[ii], performance[ii])

    return grid


# Defined overfitting plot
def plot_overfitting(estimator, X_train, X_test, y_train, y_test,
                     bins=50, pos_class=1, directory='.'):
    """
    Multi class version of Logarithmic Loss metric

    Parameters
    ----------
    estimator : array, shape = [n_samples]
         true class, intergers in [0, n_classes - 1)
    X_train : array, shape = [n_samples]
    X_test : array, shape = [n_samples]
    y_train : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_test : array, shape = [n_samples, n_classes]
    bins : int, default 50
    pos_class : int, default 1
    directory : string

    Returns
    -------
    loss : float
    """

    # Extract model name
    name = extract_estimator_name(estimator)

    # check to see if model file exist
    if not os.path.isfile(directory+'/'+str(name)+'.pkl'):
        estimator.fit(X_train, y_train)
        joblib.dump(estimator, directory+'/'+str(name)+'.pkl')

    else:
        print('Using stored model file')
        estimator = joblib.load(directory+'/'+str(name)+'.pkl')

    # use subplot to extract axis to add ks and p-value to plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Customize the major grid
    ax.grid(which='major', linestyle='-', linewidth='0.2', color='gray')
    ax.set_facecolor('white')

    # Use decision function
    if not hasattr(estimator, 'predict_proba'):
        d = estimator.decision_function(sp.vstack([X_train, X_test]))
        bin_edges_low_high = np.linspace(min(d), max(d), bins + 1)

    # use prediction function
    else:
        bin_edges_low_high = np.linspace(0., 1., bins + 1)

    label_name = ''
    y_scores = []
    for X, y in [(X_train, y_train), (X_test, y_test)]:

        if hasattr(estimator, 'predict_proba'):
            label_name = 'Prediction Probability'
            y_scores.append(estimator.predict_proba(X[y > 0])[:, pos_class])
            y_scores.append(estimator.predict_proba(X[y < 1])[:, pos_class])
        else:
            label_name = 'Decision Function'
            y_scores.append(estimator.decision_function(X[y > 0]))
            y_scores.append(estimator.decision_function(X[y < 1]))

    width = np.diff(bin_edges_low_high)

    # Signal training histogram
    hist_sig_train, bin_edges = np.histogram(y_scores[0], bins=bin_edges_low_high)

    hist_sig_train = hist_sig_train / np.sum(hist_sig_train, dtype=np.float32)

    plt.bar(bin_edges[:-1], hist_sig_train, width=width, color='r', alpha=0.5,
            label='signal (train)')

    # Background training histogram
    hist_bkg_train, bin_edges = np.histogram(y_scores[1], bins=bin_edges_low_high)

    hist_bkg_train = hist_bkg_train / np.sum(hist_bkg_train, dtype=np.float32)

    plt.bar(bin_edges[1:], hist_bkg_train, width=width,
            color='steelblue', alpha=0.5, label='background (train)')

    # Signal test histogram
    hist_sig_test, bin_edges = np.histogram(y_scores[2], bins=bin_edges_low_high)

    hist_sig_test = hist_sig_test / np.sum(hist_sig_test, dtype=np.float32)
    scale = len(y_scores[2]) / np.sum(hist_sig_test, dtype=np.float32)
    err = np.sqrt(hist_sig_test * scale) / scale

    plt.errorbar(bin_edges[:-1], hist_sig_test, yerr=err, fmt='o', c='r', label='signal (test)')

    # Background test histogram
    hist_bkg_test, bin_edges = np.histogram(y_scores[3], bins=bin_edges_low_high)

    hist_bkg_test = hist_bkg_test / np.sum(hist_bkg_test, dtype=np.float32)
    scale = len(y_scores[3]) / np.sum(hist_bkg_test, dtype=np.float32)
    err = np.sqrt(hist_bkg_test * scale) / scale

    plt.errorbar(bin_edges[:-1], hist_bkg_test, yerr=err, fmt='o', c='steelblue',
                 label='background (test)')

    ax.set_title(name, fontsize=14)

    plt.xlabel(label_name)
    plt.ylabel('Arbitrary units')

    leg = plt.legend(loc='best', frameon=False, fancybox=False, fontsize=12)
    leg.get_frame().set_edgecolor('w')

    frame = leg.get_frame()
    frame.set_facecolor('White')

    return display(plt.show())


# Define learning curve plot
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        scoring=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    scoring : string

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    train_sizes : array-like
    """

    train_sizes, train_scores, test_scores = learning_curve(
                 estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs,
                 train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Set figure size before plotting
    plt.figure(figsize=(8, 6))

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color='r')

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',
             label='Training score')

    plt.plot(train_sizes, test_scores_mean, 'o-', color='g',
             label='Cross-validation score')

    plt.title(title)

    plt.xlabel('Training sample size')
    plt.ylabel('Score: '+scoring)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.legend(loc='best')

    return display(plt.show())
