"""
============================
Underfitting vs. Overfitting
============================

This example demonstrates the problems of underfitting and overfitting and
how we can use linear regression with polynomial features to approximate
nonlinear functions. The plot shows the function that we want to approximate,
which is a part of the cosine function. In addition, the samples from the
real function and the approximations of different models are displayed. The
models have polynomial features of different degrees. We can see that a
linear function (polynomial with degree 1) is not sufficient to fit the
training samples. This is called **underfitting**. A polynomial of degree 4
approximates the true function almost perfectly. However, for higher degrees
the model will **overfit** the training data, i.e. it learns the noise of the
training data.
We evaluate quantitatively **overfitting** / **underfitting** by using
cross-validation. We calculate the mean squared error (MSE) on the validation
set, the higher, the less likely the model generalizes correctly from the
training data.
"""

print(__doc__)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectFromModel
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

degrees = [3]
a = [100]
factor=np.linspace(0.3,0.3,1)

for i in range(len(degrees)):
    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=True)

    for j in range(len(a)):
        clf = Ridge(alpha=a[j], copy_X=True, fit_intercept=True, max_iter=None,
            normalize=False,  solver='auto', tol=0.001)

        for z in range(len(factor)):
            sfm = SelectFromModel(clf, prefit=False, threshold=factor[z])
            pipeline = Pipeline([("polynomial_features", polynomial_features),
                             ("feature_selection", sfm),
                             ("Ridge", clf)])
            pipeline.fit(X, y)

 #       print("GridSearchCV took %.2f seconds for %d candidate parameter settings.")
      # (time() - start, len(grid_search.grid_scores_)))
    # Evaluate the models using crossvalidation
            print(degrees[i],a[j],factor[z])
            eval_pred(pipeline, X, y, K=3)



sub = pd.read_csv("sample.csv")
sub['y'] = pipeline.predict(X_test)
sub.head()
sub.to_csv('test_submission_ridge_poly3_selectfeature.csv', index = False)

# X_test_ridge=polynomial_features.fit_transform(X_test)
# sub = pd.read_csv("sample.csv")
# sub['y'] = pipeline.predict(X_test_ridge)
# sub.head()
# sub.to_csv('test_submission_ridge.csv', index = False)
