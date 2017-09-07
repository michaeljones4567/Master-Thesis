# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3 clause

from __future__ import print_function
from __future__ import division

import numpy as np
import code

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.isotonic import isotonic_regression
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import Lasso

from sklearn import cluster, datasets
from sklearn.metrics import pairwise_distances
from sklearn import metrics

import matplotlib.ticker as ticker

#==========================================================================================================================================================#

show_survival_graphs = 'yes'
show_covariance_matrices = 'yes'
show_heatmaps = 'yes'

Experiment = 'Full'

clustering_method = 'kmeans'

Number_of_clusters = 12


if Experiment == 'Fever':

    Number_of_clusters_to_try = 3

if Experiment == 'Cured':

    Number_of_clusters_to_try = 3

if Experiment == 'Full':

    Number_of_clusters_to_try = 3



#   Alpha Range
Start_alpha = 0.01
Final_alpha = 0.2
Alpha_interval = 0.01

#   Beta Range
Start_beta = 0.001
Final_beta = 0.02
Beta_interval = 0.001


#==========================================================================================================================================================#

def squared_loss(y_true, y_pred, return_derivative=False):
    diff = y_pred - y_true
    obj = 0.5 * np.dot(diff, diff)
    if return_derivative:
        return obj, diff
    else:
        return obj


def squared_hinge_loss(y_true, y_scores, return_derivative=False):
    # labels in (-1, 1)
    z = np.maximum(0, 1 - y_true * y_scores)
    obj = np.sum(z ** 2)

    if return_derivative:
        return obj, -2 * y_true * z
    else:
        return obj


def get_loss(name):
    losses = {'squared': squared_loss,
              'squared-hinge': squared_hinge_loss}
    return losses[name]

def fista(sfunc, nsfunc, x0, max_iter=500, max_linesearch=20, eta=2.0, tol=1e-3,
          verbose=0):

    y = x0.copy()
    x = y
    L = 1.0
    t = 1.0

    for it in range(max_iter):
        f_old, grad = sfunc(y, True)

        for ls in range(max_linesearch):
            y_proj = nsfunc(y - grad / L, L)
            diff = (y_proj - y).ravel()
            sqdist = np.dot(diff, diff)
            dist = np.sqrt(sqdist)

            F = sfunc(y_proj)
            Q = f_old + np.dot(diff, grad.ravel()) + 0.5 * L * sqdist

            if F <= Q:
                break

            L *= eta

        if ls == max_linesearch - 1 and verbose:
            print("Line search did not converge.")

        if verbose:
            print("%d. %f" % (it + 1, dist))

        if dist <= tol:
            if verbose:
                print("Converged.")
            break

        x_next = y_proj
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        y = x_next + (t-1) / t_next * (x_next - x)
        t = t_next
        x = x_next

    return y_proj

def prox_owl(v, w):
    """Proximal operator of the OWL norm dot(w, reversed(sort(v)))
    Follows description and notation from:
    X. Zeng, M. Figueiredo,
    The ordered weighted L1 norm: Atomic formulation, dual norm,
    and projections.
    eprint http://arxiv.org/abs/1409.4271
    """

    # wlog operate on absolute values
    v_abs = np.abs(v)
    ix = np.argsort(v_abs)[::-1]
    v_abs = v_abs[ix]
    # project to K+ (monotone non-negative decreasing cone)
    v_abs = isotonic_regression(v_abs - w, y_min=0, increasing=False)

    # undo the sorting
    inv_ix = np.zeros_like(ix)
    inv_ix[ix] = np.arange(len(v))
    v_abs = v_abs[inv_ix]

    return np.sign(v) * v_abs


def _oscar_weights(alpha, beta, size):
    w = np.arange(size - 1, -1, -1, dtype=np.double)
    w *= beta
    w += alpha
    return w


def _fit_owl_fista(X, y, w, loss, max_iter=500, max_linesearch=20, eta=2.0,
                   tol=1e-3, verbose=0):

    # least squares loss
    def sfunc(coef, grad=False):
        y_scores = safe_sparse_dot(X, coef)
        if grad:
            obj, lp = loss(y, y_scores, return_derivative=True)
            grad = safe_sparse_dot(X.T, lp)
            return obj, grad
        else:
            return loss(y, y_scores)

    def nsfunc(coef, L):
        return prox_owl(coef, w / L)

    coef = np.zeros(X.shape[1])
    return fista(sfunc, nsfunc, coef, max_iter, max_linesearch,
                 eta, tol, verbose)



class _BaseOwl(BaseEstimator):
    """
    Solves sum loss(y_pred, y) + sum_j weights_j |coef|_(j)
           where u_(j) is the jth largest component of the vector u.
           and weights is a monotonic nonincreasing vector.
    OWL is also known as: sorted L1 norm, SLOPE
    Parameters
    ----------
    weights: array, shape (n_features,) or tuple, length 2
        Nonincreasing weights vector for the ordered weighted L1 penalty.
        If weights = (alpha, 0, 0, ..., 0), this amounts to a L_inf penalty.
        If weights = alpha * np.ones(n_features) it amounts to L1.
        If weights is a tuple = (alpha, beta), the OSCAR penalty is used::
            alpha ||coef||_1 + beta sum_{i<j} max{|x_i|, |x_j|)
        by computing the corresponding `weights` vector as::
            weights_i = alpha + beta(n_features - i - 1)
    loss: string, default: "squared"
        Loss function to use, see loss.py to add your own.
    max_iter: int, default: 500
        Maximum FISTA iterations.
    max_linesearch: int, default: 20
        Maximum number of FISTA backtracking line search steps.
    eta: float, default: 2
        Amount by which to increase step size in FISTA bactracking line search.
    tol: float, default: 1e-3
        Tolerance for the convergence criterion.
    verbose: int, default 0:
        Degree of verbosity to print from the solver.
    References
    ----------
        X. Zeng, M. Figueiredo,
        The ordered weighted L1 norm: Atomic formulation, dual norm,
        and projections.
        eprint http://arxiv.org/abs/1409.4271
    """

    def __init__(self, weights, loss='squared', max_iter=500,
                 max_linesearch=20, eta=2.0, tol=1e-3, verbose=0):
        self.weights = weights
        self.loss = loss
        self.max_iter = max_iter
        self.max_linesearch = max_linesearch
        self.eta = eta
        self.tol = tol
        self.verbose = verbose

    def get_loss(self):
        if self.loss != 'squared':
            raise NotImplementedError('Only regression loss implemented '
                                      'at the moment is squared.')
        return get_loss(self.loss)

    def fit(self, X, y):

        n_features = X.shape[1]

        loss = self.get_loss()
        weights = self.weights
        if isinstance(weights, tuple) and len(weights) == 2:
            alpha, beta = self.weights
            weights = _oscar_weights(alpha, beta, n_features)

        self.coef_ = _fit_owl_fista(X, y, weights, loss, self.max_iter,
                                    self.max_linesearch, self.eta, self.tol,
                                    self.verbose)
        return self

    def _decision_function(self, X):
        return safe_sparse_dot(X, self.coef_)


class OwlRegressor(_BaseOwl, RegressorMixin):
    """Ordered Weighted L1--penalized (OWL) regression solved by FISTA"""
    __doc__ += _BaseOwl.__doc__

    def get_loss(self):
        if self.loss != 'squared':
            raise NotImplementedError('Only regression loss implemented '
                                      'at the moment is squared.')

        return get_loss(self.loss)

    def predict(self, X):
        return self._decision_function(X)


class OwlClassifier(_BaseOwl, ClassifierMixin):
    """Ordered Weighted L1--penalized (OWL) classification solved by FISTA"""
    __doc__ += _BaseOwl.__doc__
    def get_loss(self):
        return get_loss(self.loss)

    def fit(self, X, y):
        self.lb_ = LabelBinarizer(neg_label=-1)
        y_ = self.lb_.fit_transform(y).ravel()
        return super(OwlClassifier, self).fit(X, y_)

    def decision_function(self, X):
        return self._decision_function(X)

    def predict(self, X):
        y_pred = self.decision_function(X) > 0
        return self.lb_.inverse_transform(y_pred)


if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_boston, load_breast_cancer

    print("OSCAR proximal operator on toy example:")
    v = np.array([1, 3, 2.9, 4, 0])
    w_oscar = _oscar_weights(alpha=0.01, beta=1, size=5)
    print(prox_owl(v, w_oscar))
    print()

    print("Regression")
    X, y = load_boston(return_X_y=True)
    X = np.column_stack([X, -X[:, 0] + 0.01 * np.random.randn(X.shape[0])])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    clf = OwlRegressor(weights=(1, 100))
    clf.fit(X_tr, y_tr)
    print("Correlated coefs", clf.coef_[0], clf.coef_[-1])
    print("Test score", clf.score(X_te, y_te))
    print()

    print("Classification")
    X, y = load_breast_cancer(return_X_y=True)
    X = np.column_stack([X, -X[:, 0] + 0.01 * np.random.randn(X.shape[0])])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
    clf = OwlClassifier(weights=(1, 100), loss='squared-hinge')
    clf.fit(X_tr, y_tr)
    print("Correlated coefs", clf.coef_[0], clf.coef_[-1])
print("Test score", clf.score(X_te, y_te))

n_samples = 10
n_features = 100

coef = np.zeros(n_features)
coef[20:30] = -1
coef[60:70] = 1
coef /= np.linalg.norm(coef)

rng = np.random.RandomState(1)
X = rng.randn(n_samples, n_features)
X[:, 20:30] = X[:, 20]
X[:, 60:70] = X[:, 20]
X += 0.001 * rng.randn(n_samples, n_features)
X /= np.linalg.norm(X, axis=0)
y = np.dot(X, coef)


if Experiment == 'Cured':

    # Load training data
    Data_uncleaned = np.genfromtxt('JRTB0&2y.txt',dtype=None) # 13720, 117
    Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
    X = np.transpose(Data_cleaned) # 116, 13718
    Gene_names = Data_uncleaned[2:,0]

    # Extract data dimensions
    Number_of_data_points = np.shape(X)[0]
    Number_of_features = np.shape(X)[1]

    # Split into infected patient data and uninfected patient data
    Number_of_data_points_TB = 46
    Number_of_data_points_Clear = 31

    X_TB = X[0:Number_of_data_points_TB,:]
    X_Clear = X[Number_of_data_points_TB:Number_of_features,:]
    X = np.concatenate((X_TB, X_Clear), axis=0)
    X = X.astype(np.float)

if Experiment == 'Fever':

    # Load training data
    Data_uncleaned = np.genfromtxt('JRTB0&Fever.txt',dtype=None) # 13720, 117
    Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
    X = np.transpose(Data_cleaned) # 116, 13718
    Gene_names = Data_uncleaned[2:,0]

    # Extract data dimensions
    Number_of_data_points = np.shape(X)[0]
    Number_of_features = np.shape(X)[1]

    # Split into infected patient data and uninfected patient data
    Number_of_data_points_TB = 46
    Number_of_data_points_Clear = 70

    X_TB = X[0:Number_of_data_points_TB,:]
    X_Clear = X[Number_of_data_points_TB:Number_of_features,:]
    X = np.concatenate((X_TB, X_Clear), axis=0)
    X = X.astype(np.float)


if Experiment == 'Full':

    Data_uncleaned = np.genfromtxt('JRTB0&2y.txt',dtype=None) # 13720, 117
    Data_uncleaned_2 = np.genfromtxt('JRTB0&Fever.txt',dtype=None) # 13720, 117

    Gene_names = Data_uncleaned[2:,0]

    Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
    Data_cleaned_2 = Data_uncleaned_2[2:,1:] # 13718, 116

    Data_cleaned = np.transpose(Data_cleaned) # 116, 13718
    Data_cleaned_2  = np.transpose(Data_cleaned_2) # 116, 13718

    # Split into infected patient data and uninfected patient data
    Number_of_data_points_TB = 46
    Number_of_data_points_Clear = 101

    X_TB = Data_cleaned[0:Number_of_data_points_TB,:]
    X_Clear = Data_cleaned[Number_of_data_points_TB:,:]
    X_Fever = Data_cleaned_2[Number_of_data_points_TB:,:]

    X = np.concatenate([X_TB, X_Clear, X_Fever], axis=0)
    X = X.astype(np.float)

    # Extract data dimensions
    Number_of_data_points = np.shape(X)[0]
    Number_of_features = np.shape(X)[1]

if Experiment == 'Cured_Fever':

    Data_uncleaned = np.genfromtxt('JRTB0&2y.txt',dtype=None) # 13720, 117
    Data_uncleaned_2 = np.genfromtxt('JRTB0&Fever.txt',dtype=None) # 13720, 117

    Gene_names = Data_uncleaned[2:,0]

    Data_cleaned = Data_uncleaned[2:,1:] # 13718, 116
    Data_cleaned_2 = Data_uncleaned_2[2:,1:] # 13718, 116

    Data_cleaned = np.transpose(Data_cleaned) # 116, 13718
    Data_cleaned_2  = np.transpose(Data_cleaned_2) # 116, 13718

    # Split into infected patient data and uninfected patient data
    Number_of_data_points_TB = 31
    Number_of_data_points_Clear = 70

    X_TB = Data_cleaned[-Number_of_data_points_TB:,:]
    X_Clear = Data_cleaned_2[-Number_of_data_points_Clear:,:]

    X = np.concatenate([X_TB, X_Clear], axis=0)
    X = X.astype(np.float)

    # Extract data dimensions
    Number_of_data_points = np.shape(X)[0]
    Number_of_features = np.shape(X)[1]




# Label infected patient data and uninfected patient data
Y1 = np.ones([Number_of_data_points_TB,1])
Y2 = np.zeros([Number_of_data_points_Clear,1])
Data_Sub_Sample_Y = np.concatenate((Y1, Y2), axis=0)

Ranking_owl_pure_survival = np.zeros(np.shape(Data_uncleaned[2:,0]))
Ranking_owl_order = np.zeros(np.shape(Data_uncleaned[2:,0]))
Ranking_owl_scores = np.zeros(np.shape(Data_uncleaned[2:,0]))

counter = 0

for icount in np.arange(Start_alpha,Final_alpha,Alpha_interval):
    print(icount)
    alpha = icount

    for jcount in np.arange(Start_beta,Final_beta,Beta_interval):

        beta = jcount  # only in OWL

        oscar_owl = OwlClassifier(weights=(alpha, beta))
        oscar_owl.fit(X, Data_Sub_Sample_Y)

        scores = np.abs(oscar_owl.coef_)
        a = scores
        c = np.sort(scores)

        Ordered_args = np.argsort(scores)


        #code.interact(local=dict(globals(), **locals()))

        Ranking_owl_pure_survival[Ordered_args[-10:]] += 1
        Ranking_owl_order[Ordered_args[-10:]] += np.arange(1,11)

        Ranking_owl_scores += scores
        counter += 1


a = Data_uncleaned[2:,:]

Ranking_owl_pure_survival_args = np.argsort(Ranking_owl_pure_survival)
Ranking_owl_ordered_args = np.argsort(Ranking_owl_order)
Ranking_owl_scores_ordered_args = np.argsort(Ranking_owl_scores)

Gene_names_pure_survival = a[Ranking_owl_pure_survival_args,0]
Gene_names_ordered = a[Ranking_owl_ordered_args,0]
Gene_names_scores = a[Ranking_owl_scores_ordered_args,0]

print(Gene_names_pure_survival[-12:])
print(Gene_names_ordered[-12:])
print(Gene_names_scores[-12:])

code.interact(local=dict(globals(), **locals()))

Ranking_counter_pure_survival = Ranking_owl_pure_survival/counter
Ranking_counter_adjusted = Ranking_owl_order/(10*counter)
Top_survival_scores = Ranking_owl_scores/counter

np.save(Experiment + '_Top_10_By_Oscar',Ranking_counter_pure_survival)

np.save(Experiment + '_Top_10_Weighted_By_Oscar',Ranking_counter_adjusted)
np.save(Experiment + '_scores_By_Oscar',Top_survival_scores)
