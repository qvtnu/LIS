

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
from sklearn.feature_selection import SelectFromModel
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

clf = Ridge(alpha=a[0], copy_X=True, fit_intercept=True, max_iter=None,
            normalize=False,  solver='auto', tol=0.001)
kernel = KernelRidge(kernel='rbf', gamma=0.01, alpha=0.001)
sfm = SelectFromModel(clf, prefit=False)
krfs = Pipeline([("feature_selection", sfm),
        ("KernelRidge", kernel)])

t0 = time.time()

krfs.fit(X, y)

krfs_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % krfs_fit)
eval_pred(krfs, X, y, K=3)


# sub = pd.read_csv("sample.csv")
# sub['y'] = kernel.predict(X_test)
# sub.head()
# sub.to_csv('test_submission_kernel_poly2.csv', index = False)