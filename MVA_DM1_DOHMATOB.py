"""
:Author: DOHMATOB Elvis Dopgima
:Synopsis: source code for MVA (Introduction to Graphical Models, ENS Cachan)
DM1; needs numpy and matplotlib libraries to run

Usage
=====
python <name_of_script.py>

"""

import os
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from datasets import load_data


def matrix2tex(matrix, column_names=None, row_names=None,
               title=None, legend=None):
    """
    Converts a 2D array (i.e matrix) into a LaTex table.

    Parameters
    ----------
    matrix: 2D array of floats
        matrix to be converted to LaTex table

    column_names: string, optional (default None)
        column headers for the output LaTex table

    row_names: string, optional (default None)
        row headers for the output LaTex table

    title; string, optional (default None)
        title for the output LaTex table

    legend; string, optional (default None)
        legend for the output LaTex table

    Returns
    -------
    string: LaTex-formatted table

    """

    matrix = np.array(matrix)

    if not column_names is None:
        column_names = [" "] + column_names

    def _get_row_name(row_num):
        if row_names is None:
            return ""
        else:
            return "\\textbf{%s} & " % row_names[row_num]

    res = """
\\begin{table}
\\centering
%s
\\begin{tabular}{|l|| %s r|}
  \\hline
  %s
  %s\\\\
  \\hline
\\end{tabular}
%s
\end{table}
""" % ("\\caption{%s}" % title if not title is None else "",
       'c|' * (matrix.shape[1] - 1 - (row_names is None)),
       "    %s" % " & ".join(
        ["\\textbf{%s}" % cn for cn in column_names]
        ) + "\\\\\n\\hline\\hline\n" if not column_names is None
       else "",
       "\\\\\n\\hline\n".join([_get_row_name(line_num) + " & ".join(
                map('{0:.3f}'.format, line))
                               for line, line_num in zip(
                matrix, xrange(len(matrix)))]),
       "\\caption*{%s}" % legend if not legend is None else ""
       )

    return res


def logit(a):
    """
    Logistic sigmoid function

    """

    return 1. / (1 + np.exp(-a) + 1e-5)


class _BaseModel(object):
    """
    Base Model for all regression models implemented hereunder

    """

    def __init__(self, verbose=1):
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(msg)


class LinearReg(_BaseModel):
    """
    Linear Regression.

    """

    def _to_affine(self, X):
        return np.vstack((X.T, np.ones(X.shape[0]))).T

    def fit(self, X, Y):
        """
        Computes the ML estimates for the model parameters.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features + 1)
           design matrix, in which each row corresponds to a train sample
           point, with 1 appended to it (to correct for the intercept).
           Thus the last column corresponds to the 'constant regressor'
           in fMRI terminology.

        Y: 1D array of length n_samples
           array of target values for each row in the design matrix X

        """

        self.X_affine_ = self._to_affine(X)

        self.w_ = np.dot(np.linalg.inv(np.dot(
                    self.X_affine_.T, self.X_affine_)),
                      np.dot(self.X_affine_.T, Y))

        return self

    def predict(self, X, binary=True):
        """
        Predicts the class of a set of new input values.

        Parameters
        ----------
        X: 2D array_like  of shape (n_samples, n_features)
            new input points whose labels are sought-for

        Returns
        -------
        A 1D array of same length as there are rows in X (n_samples)

        """

        assert np.shape(X)[1] + 1 == len(self.w_)

        Y = np.dot(self._to_affine(X), self.w_)
        if binary:
            return (Y > .5)
        else:
            return Y

    def print_params(self):
        print "Estimated model parameters"
        print "=========================="
        print "w                    : %s" % self.w_


class LogitReg(_BaseModel):
    """
    Logistic Regression via Newton-Raphson gradient descent.

    """

    def _to_affine(self, X):
        """
        Returns the affine version of a design matrix X (i.e with a
        column of 1s padded to it).

        """

        return np.vstack((X.T, np.ones(X.shape[0]))).T

    def fit(self, X, Y, rtol=1e-4, atol=1e-8, max_iter=100):
        """
        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features + 1)
           design matrix, in which each row corresponds to a train sample
           point, with 1 appended to it (to correct for the intercept).
           Thus the last column corresponds to the 'constant regressor'
           in fMRI terminology.

        Y: 1D array of length n_samples
           array of target values for each row in the design matrix X

        rtol : float, optional (default 1e-4)
            the relative tolerance parameter for convergence

        atol : float, optional (default 1e-8)
            he absolute tolerance parameter for convergence

        max_iter: int, optional (default 100):
            maximum number of iterations to be run. -1 implies infinite
            computation budget (or  patience, for that matter)

        """

        self.X_affine_ = self._to_affine(X)

        # initial parameters
        w_old = np.random.randn(self.X_affine_.shape[1]) * .01

        # IRLS loop
        iter_count = 1
        converged = False
        while max_iter:
            X_dot_w = np.dot(self.X_affine_, w_old)
            sigma = logit(X_dot_w)
            R = np.diag(sigma * (1 - sigma))  # weights for RLS
            X_trans_dot_R = np.dot(self.X_affine_.T, R)

            # Newton-Raphson update
            w_new = np.dot(
                np.dot(
                    numpy.linalg.inv(np.dot(X_trans_dot_R, self.X_affine_)),
                    X_trans_dot_R),
                X_dot_w - np.dot(numpy.linalg.inv(R),
                                    sigma - Y  # error term
                                    )
                )

            # verbose
            max_iter -= 1
            iter_count += 1
            token = "\tIteration %3i: " % (iter_count) + "".join(
                ['%-12.4g ' % z for z in w_new])
            self._log(token)

            # converged ?
            if np.allclose(w_new, w_old, rtol=rtol, atol=atol):
                converged = True
                print "\tConverged!"
                break

            # move-on to next iteration
            w_old = w_new

        if not converged:
            Warning(
                ("Didn't converge after %i iterations; try increasing "
                 "max_iter, or decreasing rtol and atol"))

        self.w_ = w_old  # = w_new

        print "Learn parameter w: %s" % self.w_

        return self

    def predict(self, X):
        """
        Predicts the class of a set of new input values.

        Parameters
        ----------
        X: 2D array_like  of shape (n_samples, n_features)
            new input points whose labels are sought-for

        Returns
        -------
        A 1D array of same length as there are rows in X (n_samples)

        """

        assert np.shape(X)[1] + 1 == len(self.w_)

        return (logit(np.dot(self._to_affine(X), self.w_)) > .5)

    def print_params(self):
        print "Estimated model parameters"
        print "=========================="
        print "w                    : %s" % self.w_


class LDA(LogitReg):
    """
    Linear Discriminant Analysis.

    """

    n_classes_ = 2

    def fit(self, X, Y):
        """
        Computes the ML estimates for the model parameters.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features + 1)
           design matrix, in which each row corresponds to a train sample
           point, with 1 appended to it (to correct for the intercept).
           Thus the last column corresponds to the 'constant regressor'
           in fMRI terminology.

        Y: 1D array of length n_samples
           array of target values for each row in the design matrix X

        """

        N = len(Y)
        N1 = np.sum(Y) * 1.
        N2 = N - N1
        self.prior_ = N1 / N
        self.means_ = np.array([np.array(X)[Y == 1].mean(axis=0),
                                np.array(X)[Y == 0].mean(axis=0)])
        self.covariance_matrix_ = (N1 / N) * np.cov(np.array(X)[Y == 1].T,
                                                    bias=1) + (
            N2 / N) * np.cov(np.array(X)[Y == 0].T, bias=1)
        self.precision_ = numpy.linalg.inv(self.covariance_matrix_)

        w0 = -.5 * (np.dot(np.dot(self.means_[0], self.precision_),
                           self.means_[0]
                           ) - np.dot(np.dot(self.means_[1],
                                             self.precision_), self.means_[1]
                                      )) + np.log(self.prior_ / (
                1 - self.prior_))
        self.w_ = np.hstack((np.dot(self.precision_,
                                    self.means_[0] - self.means_[1]),
                             w0))

        return self

    def print_params(self):
        print "Estimated model parameters"
        print "=========================="
        print "Prior (pi):", self.prior_
        for k in xrange(self.n_classes_):
            print "mean of class %i      : %s" % (k + 1, self.means_[k])
        print "w                    : %s" % self.w_


class QDA(_BaseModel):
    """
    Quadratic Discriminant Analysis (multi-class)

    """

    def fit(self, X, Y):
        """
        Computes the ML estimates for the model parameters.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features + 1)
           design matrix, in which each row corresponds to a train sample
           point, with 1 appended to it (to correct for the intercept).
           Thus the last column corresponds to the 'constant regressor'
           in fMRI terminology.

        Y: 1D array of length n_samples
           array of target values for each row in the design matrix X

        """

        N = len(Y)
        self.n_classes_ = len(set(Y))
        X = np.array(X)

        # priors
        self.prior_ = [1. * (Y == k).sum() / N for k in xrange(
                self.n_classes_)]

        # means / centroids of classes
        self.means_ = [X[Y == k].mean(axis=0) for k in xrange(self.n_classes_)]

        # covariance and precision matrices of classes
        self.covariance_matrices_ = [np.cov(X[Y == k].T, bias=1)
                                     for k in xrange(self.n_classes_)]
        self.precision_matrices_ = [numpy.linalg.inv(cov)
                                    for cov in self.covariance_matrices_]

        # logarithms of determinants of covariance matrices
        self.log_det_covs_ = [np.log(numpy.linalg.det(cov))
                              for cov in self.covariance_matrices_]

        # self.print_params_in_latex()

        return self

    def _compute_qd(self, k, x):
        return -.5 * self.log_det_covs_[k] - .5 * np.dot(
            np.dot(x.T, self.precision_matrices_[k]), x) + np.log(
            self.prior_[k])

    def predict(self, X):
        """
        Predicts the class of a set of new input values.

        Parameters
        ----------
        X: 2D array_like  of shape (n_samples, n_features)
            new input points whose labels are sought-for

        Returns
        -------
        A 1D array of same length as there are rows in X (n_samples)

        """

        X = np.array(X)

        qd = np.ndarray((len(X), self.n_classes_))
        for k in xrange(self.n_classes_):
            for i in xrange(len(X)):
                x = X[i] - self.means_[k]
                qd[i, k] = self._compute_qd(k, x)

        return np.argmax(qd, axis=1)

    def print_params(self):
        print "Estimated model parameters"
        print "=========================="
        print "Prior (pi)                   :", self.prior_
        for k in xrange(self.n_classes_):
            print "mean of class %i              : %s" % (k + 1,
                                                          self.means_[k])
            print "covariance matrix of class %i : %s" % (
                k + 1, self.covariance_matrices_[k])

    def print_params_in_latex(self):
        print "\\["
        for mean, i in zip(self.means_,
                          xrange(len(self.means_))):
            print """\\hat{\\mu}_{%i} =
  \\begin{pmatrix}
    %s
  \\end{pmatrix}, """ % (i + 1, " & ".join([str(r) for r in mean]))

        print "\\]"
        print "\\["
        for cov, i in zip(self.covariance_matrices_,
                          xrange(len(self.covariance_matrices_))):
            print """\\hat{\\Sigma}_{%i} =
  \\begin{pmatrix}
    %s
  \\end{pmatrix}, """ % (i + 1,
                         "\\\\".join([" & ".join(
                  [str(c) for c in r]) for r in cov]))
        print "\\]"


if __name__ == '__main__':
    url_pattern = (
        'http://www.di.ens.fr/~fbach/courses/fall2013/classification%s.%s'
        )
    datasets = []
    for dataset_id in 'ABC':
        datasets.append([url_pattern % (dataset_id, train_or_test)
                         for train_or_test in ['train', 'test']])

    models = ['LDA', 'LogitReg', 'LinearReg', 'QDA']
    n_models = len(models)
    n_datasets = len(datasets)
    scores = np.ndarray((n_datasets * 2, n_models))
    for dataset, j in zip(datasets, xrange(n_datasets)):
        train, test = dataset

        # load the data
        dataset_name = os.path.basename(train.replace(".train", ""))
        print ">" * 80, "Begin (%s)" % dataset_name
        train = load_data(train, data_dir=os.getcwd())
        test = load_data(test, data_dir=os.getcwd())
        X_train = train[..., :-1]
        Y_train = train[..., -1]
        X_test = test[..., :-1]
        Y_test = test[..., -1]

        # fit models
        for model_id, i in zip(models, xrange(n_models)):
            print "\nRunning %s ..." % model_id
            model = eval(model_id)(verbose=0).fit(X_train, Y_train)
            print "... done."
            model.print_params()

            mistrain = (model.predict(X_train) != Y_train
                        ).sum() / (1. * len(Y_train))
            mistest = (model.predict(X_test) != Y_test
                       ).sum() / (1. * len(Y_test))
            scores[2 * j, i] = mistrain
            scores[2 * j + 1, i] = mistest
            print "Misclassification rate on train data : %.6f" % (
                mistrain)
            print "Misclassification rate on test data : %.6f" % mistest

            print "Generating plots for %s (this may take a while) ..." % (
                model_id)
            ax = plt.subplot2grid((n_datasets, n_models), (j, i))

            if j == 0:
                ax.set_title(models[i])

            if i == 0:
                ax.set_ylabel('classification%s' % 'ABC'[j])

            # plot the decision boundary. For that, we will assign a color
            # to each point in the mesh [x_min, m_max]x[y_min, y_max].
            h = .05  # step size in the mesh; small values give better figures

            x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
            y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                    np.arange(y_min, y_max, h))
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

            # put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.hold('on')
            ax.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

            ax.scatter(X_train[:, 0], X_train[:, 1], c=Y_train, edgecolors='k',
                       cmap=plt.cm.Paired)

            ax.scatter(X_test[:, 0], X_test[:, 1], c=Y_test, edgecolors='k',
                       cmap=plt.cm.Paired)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            plt.xticks(())
            plt.yticks(())

            print "... done."
        print "<" * 80, "End (%s)" % dataset_name
        print

    # save tables and figures for latex report
    print "Saving tables and figures ..."
    open("_vs_".join(models) + ".tex", "w").write(matrix2tex(
            scores, column_names=models,
            row_names=[os.path.basename(x)
                       for dataset in datasets for x in dataset],
            title=" vs ".join(models)))

    plt.savefig("%s_figures.png" % "_".join(models),
                bbox_inches="tight", dpi=200)
    print "... done."

    plt.show()
