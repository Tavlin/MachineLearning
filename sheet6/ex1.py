import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import numpy as np
import math

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def one_vs_rest():

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    lg_ovr   = LogisticRegression(multi_class='ovr', solver='lbfgs', max_iter=100).fit(X_train, y_train)
    lg_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100).fit(X_train, y_train)

    x_plot_ovr = [[],[],[]] # predicted class1 class2 class 3
    t_plot_ovr = [[],[],[]]

    x_plot_multi = [[],[],[]] # predicted class1 class2 class 3
    t_plot_multi = [[],[],[]]

    #t_color = [50., 55., 60.] # tru class1 class2 class3
    t_color = ['xkcd:maroon', 'xkcd:blurple', 'xkcd:evergreen'] # tru class1 class2 class3

    t_train = []

    for t in y_train:
        t_train.append(t_color[t])

    for it in range(0, len(y_test)):

        # -- one versus rest --

        probabilities = lg_ovr.predict_proba([X_test[it]])[0]
        max_index = np.argmax(probabilities)

        x_plot_ovr[max_index].append(X_test[it])
        t_plot_ovr[max_index].append(t_color[y_test[it]])

        # -- multinomial --

        probabilities = lg_multi.predict_proba([X_test[it]])[0]
        max_index = np.argmax(probabilities)

        x_plot_multi[max_index].append(X_test[it])
        t_plot_multi[max_index].append(t_color[y_test[it]])

    ## ------ Plotting -----------

    class1_patch = mpatches.Patch(color=t_color[0], label='true class 1')
    class2_patch = mpatches.Patch(color=t_color[1], label='true class 2')
    class3_patch = mpatches.Patch(color=t_color[2], label='true class 3')
    none_patch = mpatches.Patch(color='white')
    d_line = mlines.Line2D([], [], color='k', marker='d', markersize=7, label='predicted class 1')
    x_line = mlines.Line2D([], [], color='k', marker='X', markersize=7, label='predicted class 2')
    o_line = mlines.Line2D([], [], color='k', marker='o', markersize=7, label='predicted class 3')

    sternchen_line = mlines.Line2D([], [], color='k', marker='*', markersize=7, label='training data')

    plt.figure(0)

    scatter_ovr_class1 = plt.scatter(np.take(x_plot_ovr[0], 2, axis=1), np.take(x_plot_ovr[0], 3, axis=1),
                                     c=t_plot_ovr[0], marker='d', cmap='tab10', facecolor='none')
    scatter_ovr_class2 = plt.scatter(np.take(x_plot_ovr[1], 2, axis=1), np.take(x_plot_ovr[1], 3, axis=1),
                                     c=t_plot_ovr[1], marker='X', cmap='tab10', facecolor='none')
    scatter_ovr_class3 = plt.scatter(np.take(x_plot_ovr[2], 2, axis=1), np.take(x_plot_ovr[2], 3, axis=1),
                                     c=t_plot_ovr[2], marker='o', cmap='tab10', facecolor='none')

    scatter_train_ovr = plt.scatter(np.take(X_train, 2, axis=1), np.take(X_train, 3, axis=1),
                                        c=t_train, marker='*', cmap='tab10', alpha=0.3)

    plt.legend(handles=[class1_patch, class2_patch, class3_patch, none_patch, d_line, x_line, o_line, none_patch,
                        sternchen_line])
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.suptitle("three-class classification, one versus rest")
    plt.savefig("one_versus_rest.png", dpi=300, format="png")

    plt.figure(1)

    scatter_multi_class1 = plt.scatter(np.take(x_plot_multi[0], 2, axis=1), np.take(x_plot_multi[0], 3, axis=1), \
                                     c=t_plot_multi[0], marker='d', cmap='tab10', facecolor='none')
    scatter_multi_class2 = plt.scatter(np.take(x_plot_multi[1], 2, axis=1), np.take(x_plot_multi[1], 3, axis=1), \
                                     c=t_plot_multi[1], marker='X', cmap='tab10', facecolor='none')
    scatter_multi_class3 = plt.scatter(np.take(x_plot_multi[2], 2, axis=1), np.take(x_plot_multi[2], 3, axis=1), \
                                     c=t_plot_multi[2], marker='o', cmap='tab10', facecolor='none')
    scatter_train_multi = plt.scatter(np.take(X_train, 2, axis=1), np.take(X_train, 3, axis=1),
                                     c=t_train, marker='*', cmap='tab10', alpha=0.3)

    plt.legend(handles=[class1_patch, class2_patch, class3_patch, none_patch, d_line, x_line, o_line, none_patch,
                        sternchen_line])
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.suptitle("three-class classification, multinomial")
    plt.savefig("multinomial.png", dpi=300, format="png")

    plt.show()

def multinomial():
    pass

def main():
    one_vs_rest()
    multinomial()

if __name__ == "__main__":
    main()