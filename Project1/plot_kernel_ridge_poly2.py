"""
=============================================
Comparison of kernel ridge regression and SVR
=============================================

Both kernel ridge regression (KRR) and SVR learn a non-linear function by
employing the kernel trick, i.e., they learn a linear function in the space
induced by the respective kernel which corresponds to a non-linear function in
the original space. They differ in the loss functions (ridge versus
epsilon-insensitive loss). In contrast to SVR, fitting a KRR can be done in
closed-form and is typically faster for medium-sized datasets. On the other
hand, the learned model is non-sparse and thus slower than SVR at
prediction-time.

This example illustrates both methods on an artificial dataset, which
consists of a sinusoidal target function and strong noise added to every fifth
datapoint. The first figure compares the learned model of KRR and SVR when both
complexity/regularization and bandwidth of the RBF kernel are optimized using
grid-search. The learned functions are very similar; however, fitting KRR is
approx. seven times faster than fitting SVR (both with grid-search). However,
prediction of 100000 target values is more than tree times faster with SVR
since it has learned a sparse model using only approx. 1/3 of the 100 training
datapoints as support vectors.

The next figure compares the time for fitting and prediction of KRR and SVR for
different sizes of the training set. Fitting KRR is faster than SVR for medium-
sized training sets (less than 1000 samples); however, for larger training sets
SVR scales better. With regard to prediction time, SVR is faster than
KRR for all sizes of the training set because of the learned sparse
solution. Note that the degree of sparsity and thus the prediction time depends
on the parameters epsilon and C of the SVR.
"""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# License: BSD 3 clause


from __future__ import division
import time

import numpy as np

from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt


print(__doc__)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
#from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.stats import sem
import math
import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn import grid_search
# Contains linear models, e.g., linear regression, ridge regression, LASSO, etc.
import sklearn.linear_model as sklin
# Provides train-test split, cross-validation, etc.
import sklearn.cross_validation as skcv
# Provides grid search functionality
import sklearn.grid_search as skgs
# For data normalization
import sklearn.preprocessing as skpr
# Allows us to create custom scoring functions
import sklearn.metrics as skmet


def score(gtruth, pred):
    diff = gtruth - pred
    return np.sqrt(np.mean(np.square(diff)))


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def eval_pred(predictor, X, y, K=3):
    #ret = {}
    #scores = cross_val_score(pred, X, y, cv = 3)
    #print (scores)
    #print ("CV score: %.4f +/- %.4f" % (np.mean(scores), sem(scores)))
    scorefun = skmet.make_scorer(score)
    scores = skcv.cross_val_score(predictor, X, y, scoring=scorefun, cv=10)
    print('C-V score =', np.mean(scores), '+/-', np.std(scores))

X = pd.read_csv('train.csv')

del X['Id']
y = X.y
del X['y']
pd.DataFrame(X).head()
X_test = pd.read_csv('test.csv')
del X_test['Id']

print(X.shape, y.shape, X_test.shape)
#############################################################################
# Fit regression model
train_size = 900

degrees = 2
a = [100]

polynomial_features = PolynomialFeatures(degree=degrees,
                          include_bias=True)
kernel = KernelRidge(kernel='rbf', gamma=0.01,alpha=0.001)
kr = Pipeline([("polynomial_features", polynomial_features),
        ("KernelRidge", kernel)])

t0 = time.time()
kr.fit(X, y)
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % kr_fit)
eval_pred(kr, X, y, K=3)


# sub = pd.read_csv("sample.csv")
# sub['y'] = kernel.predict(X_test)
# sub.head()
# sub.to_csv('test_submission_kernel_poly2.csv', index = False)