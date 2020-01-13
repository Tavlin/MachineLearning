import matplotlib.pyplot as plt
import numpy as np
import math

def basis_functions(v, a, b):

    phi = []

    for x in v:
        phi.append([1, a[0]*x[0] + b[0], a[1]*x[1]+b[1]])

    return phi

def sigmoid(a):

    return 1./ (1 + math.exp(-a))

def y_vec(w, phi):

    y = []

    for phi_n in phi:

        y.append(sigmoid(np.dot(w, phi_n)))

    return y

def special_dot(vec, vec_of_vec):

    vec.tolist()
    summi = np.array([0., 0., 0.])

    for it in range(0, len(vec)):

        summi += [vec_of_vec[it][0]*vec[it], vec_of_vec[it][1]*vec[it], vec_of_vec[it][2]*vec[it]]

    return summi


def two_class_classification():

    np.random.seed()

    # -- define program specific parameters --

    set_size = 100

    mu_1 = [1, 2]
    mu_2 = [3, 3]

    sigma_1 = [[0.5, 0.3], [0.3, 0.5]]
    sigma_2 = [[0.7, -0.3], [-0.3, 0.5]]

    x_1 = np.random.multivariate_normal(mu_1, sigma_1, size=set_size)
    x_2 = np.random.multivariate_normal(mu_2, sigma_2, size=set_size)

    v = np.append(x_1, x_2, axis = 0)
    t = np.append(np.full(set_size, 0), np.full(set_size, 1))

    # -- set up classification --

    alpha = .03

    phi = basis_functions(v, [.1, .2], [0, 1])
    w = np.full(3, 1) ## guess weight
    w_old = 0

    while np.linalg.norm(w - w_old) > 0.01:

        w_new = w - alpha*special_dot(y_vec(w, phi) - t, phi)
        w_old = w
        w = w_new

        print(np.linalg.norm(w - w_old))

    class_1 = []
    t_1 = []

    class_2 = []
    t_2 = []

    it = 0

    for phi_n in phi:

        p_class_1 = np.dot(w, phi_n)

        if p_class_1 > 0.5:
            class_1.append(phi_n)
            t_1.append(.1 if t[it] == 0 else .9)
        else:
            class_2.append(phi_n)
            t_2.append(.1 if t[it] == 0 else .9)

        it += 1

    #plt.scatter(np.take(phi, 1, axis=1), np.take(phi, 2, axis=1), c=t)
    scatter1 = plt.scatter(np.take(class_1, 1, axis=1), np.take(class_1, 2, axis=1), c=t_1, marker='*', cmap='RdBu')
    scatter2 = plt.scatter(np.take(class_2, 1, axis=1), np.take(class_2, 2, axis=1), c=t_2, marker='^', cmap='RdBu')
    boundary = plt.plot([0, -w[0]/w[1]], [-w[0]/w[2], 0], 'k-')

    handles1, labels1 = scatter1.legend_elements(prop="colors", alpha=0.6)
    handles2, labels2 = scatter2.legend_elements(prop="colors", alpha=0.6)

    handles = handles1 + handles2 + boundary
    labels = ['false class 1', 'true class 1', 'true class 2', 'false class 2', 'decision boundary']

    plt.legend(handles, labels, loc="upper right")
    plt.suptitle("two-class classification")

    plt.xlabel(r'$\Phi_1(x_1)$')
    plt.ylabel(r'$\Phi_2(x_2)$')

    plt.savefig("decision_boundary.png", dpi=300, format="png")
    plt.show()


def main():
    two_class_classification()

if __name__ == "__main__":
    main()