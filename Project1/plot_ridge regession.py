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
from sklearn import cross_validation
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
#from sklearn.cross_validation import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.stats import sem
import math
import numpy as np
import matplotlib.pyplot as plt

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

degrees = [2]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=True)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X, y)

    # Evaluate the models using crossvalidation

    scores = cross_validation.cross_val_score(pipeline,
        X, y, scoring="r2", cv=5)

    print('C-V score =', np.mean(scores), '+/-', np.std(scores))


X_test_poly2=polynomial_features.fit_transform(X_test)
sub = pd.read_csv("sample.csv")
sub['y'] = pipeline.predict(X_test_poly2)
sub.head()
sub.to_csv('test_submission_poly2.csv', index = False)


degrees = [2]

plt.figure(figsize=(14, 5))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=True)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X, y)

    # Evaluate the models using crossvalidation

    scores = cross_validation.cross_val_score(pipeline,
        X, y, scoring="r2", cv=5)

    print('C-V score =', np.mean(scores), '+/-', np.std(scores))


X_test_poly2=polynomial_features.fit_transform(X_test)
sub = pd.read_csv("sample.csv")
sub['y'] = pipeline.predict(X_test_poly2)
sub.head()
sub.to_csv('test_submission_poly2.csv', index = False)

#predict =
#     plt.plot(X_test, pipeline.predict(X_test), label="Model")
#     plt.scatter(X[1:100, 1], y[1:100], label="Samples")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.xlim((-1, 3))
#     plt.ylim((-100,100))
#     plt.legend(loc="best")
#     plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
#         degrees[i], -scores.mean(), scores.std()))
# plt.show()
