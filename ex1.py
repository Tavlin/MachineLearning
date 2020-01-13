import matplotlib.pyplot as plt
import numpy as np

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
        phi_trans = np.array(np.exp(-1 / (2. * self.sigma ** 2) * phi_trans.T ** 2))
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
        #print(X)
        #print(self.steps[-1].weights_)
        return X #self

    def predict(self, X):

        for step in self.steps[:-1]:
            X = step.transform(X)
        return self.steps[-1].predict(X), X

def prediction():

    np.random.seed()

    # -- define program specific parameters --

    N_basis = 9
    Nssss = [2, 4, 10, 100]
    N_prediction = 100

    alpha = 2.0
    epsilon = 0.3
    beta = (1 / epsilon) ** 2

    lam = alpha / beta

    # -- set up prediction --

    x_pred = np.linspace(-0.0, 1., N_prediction)
    f = np.sin(2 * np.pi * x_pred)

    ssss = [0.3, 0.2, 0.1]

    for s in ssss:

        plt.figure(2*ssss.index(s), figsize=(10,6))
        plt.subplots_adjust(left=None, bottom=.15, right=None, top=None, wspace=None, hspace=.35)

        plt.suptitle("predictive distribution for sigma = " + s.__str__())

        plt.figure(2*ssss.index(s) + 1, figsize=(10,6))
        plt.subplots_adjust(left=None, bottom=.15, right=None, top=None, wspace=None, hspace=.35)

        plt.suptitle("samples drawn from predictive distribution for sigma = " + s.__str__())

        for N in Nssss:

            x_in = np.random.uniform(size=N)

            ttt = np.sin(2 * np.pi * x_in) + np.random.normal(loc=0, scale=1.0, size=N) * epsilon

            nlm = Pipeline([GaussianBasisFunctions(np.linspace(0, 1, N_basis)[:, None], sigma=s),
                            InterceptFeature(),  # add constant basis function
                            LinearRegression(lam)])
            X = nlm.fit(x_in[:, None], ttt)

            # Show training data and model predictions
            t_prediction, X_pred = nlm.predict(x_pred[:, None])

            # -- compute predictive m_N and S_N^-1 --

            S_N_inv = alpha*np.identity(N_basis+1) + beta*np.dot(X.T, X)
            S_N = np.linalg.inv(S_N_inv)

            m_N = np.linalg.solve(S_N_inv, beta*np.dot(X.T, ttt))
            mean = np.dot(m_N.T, X_pred.T)

            sigma_N_squared = []

            for line in X_pred:

                sigma_N_squared.append(1./beta + np.dot(line.T, np.dot(S_N, line)))

            sigma_N = np.sqrt(sigma_N_squared)

            # ------- PLOTTING --------

            # - predictive distribution -
            plt.figure(2*ssss.index(s))
            plt.subplot(221 + Nssss.index(N))

            plt.plot(x_in, ttt, 'b.', fillstyle='none')
            plt.plot(x_pred, f, 'g-')
            plt.plot(x_pred, mean, 'r-')
            plt.fill_between(x_pred, mean - sigma_N, mean + sigma_N, facecolor='red', alpha=0.3)
            plt.xlabel('x')
            plt.ylabel('t')

            # - samples -
            plt.figure(2*ssss.index(s) + 1)
            plt.subplot(221 + Nssss.index(N))

            plt.plot(x_in, ttt, 'b.', fillstyle='none')
            plt.plot(x_pred, f, 'g-')

            for it in range(0, 5):

                w = np.random.multivariate_normal(m_N, S_N)
                t_sample = np.dot(X_pred, w)

                plt.plot(x_pred, t_sample, 'r-')

            plt.xlabel('x')
            plt.ylabel('t')

        plt.figure(2*ssss.index(s))
        plt.figlegend(['target data', 'true function', 'predictive mean', 'predictive standard deviation'])
        plt.savefig("predictive_distribution_s=" + s.__str__() +  ".png", dpi=300, format="png")

        plt.figure(2*ssss.index(s) + 1)
        plt.figlegend(['target data', 'true function', 'prediction'])
        plt.savefig("predictive_distribution__samples_s=" + s.__str__() + ".png", dpi=300, format="png")
        plt.show()


def main():
    prediction()

if __name__ == "__main__":
    main()