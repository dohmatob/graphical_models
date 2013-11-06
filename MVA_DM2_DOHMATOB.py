"""
:Author: DOHMATOB Elvis Dopgima

"""

import numpy as np
from scipy import linalg
from datasets import load_data


def _normal_pdf(mean, cov, x):
    """
    Computes the pdf at the point x, of a multivariate normal distribution
    with given mean and covariance.

    """

    y = np.array(x - mean).reshape((-1, 1))

    return (1. / np.sqrt(linalg.det(cov) * (2 * np.pi) ** len(y))) * np.exp(
        -.5 * y.T.dot(linalg.inv(cov).dot(y)))


class KMeans(object):
    """
    KMeans algorithm for unsupervised clustering and classfication.

    Parameters
    ----------
    n_classes: int
        number of classes

    init: string, optional (default "random")
        initialization mode for the learner/model

    verbose: int, optional (default 1)
        verbosity level for logging. A value of 0 means "No verbose!"

    See Also
    --------
    EM (Expectation Maximization)

    """

    def __init__(self, n_classes, init="random", verbose=1):
        self.init = init
        self.n_classes = n_classes
        self.verbose = verbose

    def _log(self, msg, level=0):
        """
        Logs message (msg) if the current verbose level is greater than level.

        """

        if self.verbose > level:
            print msg

    def fit(self, X, max_iter=100):
        """
        Learns a hard-assignment of labels to points in a train set X.

        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            training data

        max_iter: int, optional (default 100)
            maximum number of iterations were are ready to go

        Returns
        -------
        self; KMeans
            learned KMeans model

        """

        # initialize labels
        self._log("Initializing %i cluster centroids..." % self.n_classes)
        if self.init == "random":
            self.centroids_ = X[np.random.permutation(range(
                        X.shape[0]))[:self.n_classes]]
        else:
            raise NotImplementedError(
                "Unimplemented K-Means initialization method: %s" % self.init)
        self._log("... done.")

       # loop
        self._log("Starting main loop (%i iterations)..." % max_iter)

        old_labels = np.zeros(X.shape[0])
        for iteration in xrange(max_iter):
            self._log("Iteration %i/%i..." % (iteration + 1, max_iter))

            # re-compute distances and labels for samples that changed their
            # class over the last iteration
            self.labels_ = np.argmin([[linalg.norm(x - c)
                                       for c in self.centroids_]
                                      for x in X], axis=1)

            # converged ?
            if np.all(self.labels_ == old_labels):
                self._log(" ... done; converged.")
                break

            # re-compute centroids
            for c in xrange(self.n_classes):
                self.centroids_[c] = np.mean(X[self.labels_ == c])

            old_labels = np.array(self.labels_)

        if iteration == max_iter - 1:
            self._log(
                "...done; did not converge after %i iterations." % max_iter
                )

        # return fitted object
        return self


class EM(object):
    """
    Expectation Maximization algorithm for unsupervised clustering and
    classfication.

    Parameters
    ----------
    n_classes: int
        number of classes

    verbose: int, optional (default 1)
        verbosity level for logging. A value of 0 means "No verbose!"

    See Also
    --------
    KMeans

    """

    def __init__(self, n_classes, verbose=1):
        self.n_classes = n_classes
        self.verbose = verbose

    def _log(self, msg, level=0):
        """
        Logs message (msg) if the current verbose level is greater than level.

        """

        if self.verbose > level:
            print msg

    def fit(self, X, max_iter=100):
        """
        Parameters
        ----------
        X: 2D array of shape (n_samples, n_features)
            training data

        max_iter: int, optional (default 100)
            maximum number of iterations were are ready to go

        Returns
        -------
        self: EM object
            learned EM model

        """

        X = np.array(X)
        N = X.shape[0]

        # fit KMeans model to warmstart
        self._log("Fitting KMeans for warmstarting EM...")
        km = KMeans(self.n_classes, verbose=self.verbose).fit(
            X, max_iter=max_iter)
        self._log("... done (KMeans).")

        # initialize means
        self.means_ = km.centroids_

        # labels (soft)
        self.labels_ = np.zeros((self.n_classes, N))
        for k in xrange(self.n_classes):
            self.labels_[k, km.labels_ == k] = 1

        # intitialize convariance matrices
        self.covariance_matrices_ = np.array([np.cov(X[km.labels_ == k].T,
                                                     bias=1)
                                              for k in xrange(self.n_classes)])

        # priors (mixing proportions)
        self.priors_ = np.array([1. * len(np.nonzero(
                        km.labels_ == k)[0]) / N
                                 for k in xrange(self.n_classes)])

        for iteration in xrange(max_iter):
            print "Iteration %i/%i..." % (iteration + 1, max_iter)

            ###########
            # E-step
            for n in xrange(N):
                normalizer = np.sum([1. * self.priors_[h] * _normal_pdf(
                            self.means_[h], self.covariance_matrices_[h],
                            X[n]) for h in xrange(self.n_classes)])
                for k in xrange(self.n_classes):
                    self.labels_[k, n] = 1. * self.priors_[k] * _normal_pdf(
                        self.means_[k], self.covariance_matrices_[k],
                        X[n]) / normalizer

            ###########
            # M-step
            self.means_ = self.labels_.dot(X)  # uncorrected
            counts = self.labels_.sum(axis=1)  # number of points in each class
            for k in xrange(self.n_classes):
                # update proportions
                self.priors_[k] = counts[k] / N

                # update means
                self.means_[k] /= counts[k]

                # update covariance matrices
                X_k = X - self.means_[k]
                D = np.diag(self.labels_[k])
                self.covariance_matrices_[k] = X_k.T.dot(D.dot(
                        X_k)) / counts[k]

        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = load_data(
        "http://www.di.ens.fr/~fbach/courses/fall2013/EMGaussian.data")

    # fit
    em = EM(4).fit(X, max_iter=50)

    # plot results
    plt.scatter(*X.T, c=em.labels_.argmax(axis=0))

    plt.show()
