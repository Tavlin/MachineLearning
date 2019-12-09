import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import linear_model
import numpy as np
import math
import time


class LinearRegression:
    """
    Simple linear regression
    """

    def __init__(self, lam):
        """
        Create a linear regression model
        t = X w + noise

        that minimizes
        ||X w - t||_2
        """
        self.weights_ = None
        self.lambda_ = lam

    def fit(self, X, t):
        """
        Fit linear model on training data D = (X, t)
        """
        num_samples, num_features = X.shape
        self.weights_ = np.linalg.solve(np.dot(X.T, X) + self.lambda_ * np.identity(num_features),
                                        np.dot(X.T, t))

        return self

    def predict(self, X):
        """
        Predict model response on inputs X
        """
        num_samples, num_features = X.shape

        return np.dot(X, self.weights_)

class InterceptFeature:
    """
    Constant intercept feature
    """
    def transform(self, x):
        return np.hstack([np.ones((x.shape[0], 1)), x])


class GaussianBasisFunctions():
    """
    Transform the input with a gaussian function of the form:
    phi(x) = exp(- kernelsize(=500) * (x - mu) ** 2)
    """

    def __init__(self, mus, sigma=1.0):
        self.mus = mus
        self.sigma = sigma

    def transform(self, X):
        phi = np.repeat(X, np.shape(self.mus)[0], axis=1)
        phi_trans = phi.T - self.mus
        phi_trans = np.array(np.exp(- 500.0 / self.sigma ** 2.0 * phi_trans.T ** 2.0))
        return phi_trans

class Pipeline:
    """
    Model pipeline of preprocessing steps and actual model
    """
    def __init__(self, steps):

        self.steps = steps

    def fit(self, X, y):

        for step in self.steps[:-1]:
            X = step.transform(X)
        # Last step of pipeline is actual model
        self.steps[-1].fit(X, y)
        return self

    def predict(self, X):

        for step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1].predict(X)


def bias_variance_decomp_3_5():

    np.random.seed(42)

    N = 25
    N_basis = 24
    N_prediction = 100
    x = np.random.uniform(size=N)
    y = np.linspace(0, 1, N)
    h = np.sin(2*np.pi*y)

    x_pred = np.linspace(-0.0, 1., N_prediction)

    loglambdas = [-2.4, -0.32, 2.6]

    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(left=None, bottom=.15, right=None, top=None, wspace=None, hspace=.35)

    for loglam in loglambdas:

        m = np.zeros(N_prediction)
        plt.subplot(321 + 2*loglambdas.index(loglam))

        for it in range(0, 100, 1):

            t = np.sin(2*np.pi*x) + np.random.normal(loc=0, scale=1.0, size=N)*0.3

            lam = math.exp(loglam)

            nlm = Pipeline([GaussianBasisFunctions(np.linspace(0, 1, N_basis)[:, None], sigma=1.),
                            InterceptFeature(),  # add constant basis function
                            LinearRegression(lam)])
            nlm.fit(x[:, None], t)
            # Show training data and model predictions

            y_pred = nlm.predict(x_pred[:, None])

            m += y_pred

            if (it % 5 == 0):
                plt.plot(x_pred, y_pred, 'r-', linewidth=0.5)

        m /= 100

        plt.legend([r'$\lambda =$'+loglam.__str__()], fontsize=10)
        plt.ylim(-1.5, 1.5)
        plt.xlabel('x', fontsize=15)
        if loglambdas.index(loglam) == 1:
            plt.ylabel('prediction', fontsize=15)
        plt.plot(y, h, 'k-')

        plt.subplot(322 + 2*loglambdas.index(loglam))

        plt.plot(x_pred, m, 'b-')
        plt.plot(y, h, 'k-')
        plt.ylim(-1.5, 1.5)
        plt.xlabel('x', fontsize=15)
        #plt.ylabel('', fontsize=15)
        plt.legend([r'average', r'truth'], fontsize=10)
        plt.plot(y, h, 'k-')

    plt.suptitle('Linear Regression')

    plt.savefig("linear_regression.png", dpi=300, format="png")
    plt.show()


def bias_variance_decomp_3_6():

    np.random.seed(42)

    N = 25
    N_basis = 24
    N_prediction = 100

    x = np.random.uniform(size=N)
    x = np.sort(x)

    h = np.sin(2 * np.pi * x)

    loglambdas = np.linspace(-2.4, 1.6, 100)
    bias_squared = []
    variance = []
    noise = []
    test_error = []

    for loglam in loglambdas:

        t_sum = 0

        for it in range(0, N_prediction, 1):

            t = np.sin(2*np.pi*x) + np.random.normal(loc=0, scale=1., size=N)*0.3

            lam = math.exp(loglam)

            nlm = Pipeline([GaussianBasisFunctions(np.linspace(0, 1, N_basis)[:, None], sigma=1.),
                            InterceptFeature(),  # add constant basis function
                            LinearRegression(lam)])
            nlm.fit(x[:, None], t)
            # Show training data and model predictions

            y_pred = nlm.predict(x[:, None])

            if it == 0:
                tt = y_pred[:, None]

            else:
                tt = np.hstack([tt, y_pred[:,None]])

            t_sum += np.dot(t - h, t - h)

        m = np.mean(tt, axis=1)

        bias_squared.append(np.dot(m - h, m - h) / N)
        variance.append(np.mean(np.var(tt, axis=1), axis=0))

        noise.append(t_sum / (N*N_prediction))

    plt.figure(figsize=(6, 5))
    plt.plot(loglambdas, bias_squared, 'b-')
    plt.plot(loglambdas, variance, 'r-')
    plt.plot(loglambdas, np.add(variance, bias_squared), 'm-')
    plt.plot(loglambdas, np.add(np.add(variance, bias_squared), noise), 'k-')

    plt.ylim(-0.1, 0.25)
    plt.xlabel(r'ln $\lambda$', fontsize=15)
    plt.legend([r'(bias)$^2$', r'variance', r'(bias)$^2$ + variance', 'test error'], fontsize=10)

    plt.savefig("error_curve.png", dpi=300, format="png")
    plt.show()


def lasso_test():
    np.random.seed(time.localtime())
    N = 25 # number of samples in the target
    M = 8 # number of features
    clf = linear_model.Lasso(alpha=0.1)
    x = np.random.uniform(size=(N, M))  # creates an array with dim [N][M]
    # t = some linear function(x1,...xM)
    t = 0.1 * x[:, 0] + 0.5 * x[:, 1] - 7.0 * x[:, 2] - 2.0 * x[:, 3] \
        + 1.4 * x[:, 4] + 3.3 * x[:, 5] - 0.08 * x[:, 6] - 1.1 * x[:, 7]
    alphas_lasso, coefs_lasso, dual_gaps = clf.path(x, t)   # throw out the lasso. YEAH.

    # Plotting part of the code
    plt.figure(1)
    colors = cycle(['b', 'r', 'g', 'c', 'k'])  # needs "from itertools import cycle"
    log_alphas_lasso = np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(log_alphas_lasso, coef_l, c=c)

    plt.xlabel(r'ln$\alpha$', fontsize=10)
    plt.ylabel('coefficients', fontsize=10)
    plt.savefig("Lasso.png", dpi=300, format="png")
    plt.clf()

def main():
    bias_variance_decomp_3_5()
    bias_variance_decomp_3_6()
    lasso_test()

if __name__ == "__main__":
    main()