#!/bin/python

import helpers

import numpy as np
import pyhhmm.utils
import sys
import matplotlib.pyplot as plt

if __name__ == '__main__':
    model_filename = sys.argv[1]

    model = pyhhmm.utils.load_model(model_filename)

    print("Start probabilities:\n")
    print(model.pi)
    print("\nTransition probabilities:\n")
    print(model.A)
    print("\nMeans:\n")
    print(model.means)
    print("\nCovariance matrices:\n")
    print(model.covars)
    sys.stdout.flush()

    # any 3d plot libraries? would be helpful to see the joint probability dists for pairs of features
    f = helpers.gaussian_hinton_diagram(
        model.pi,
        model.A,
        model.means[:,0].ravel(),
        [x[0][0] for x in model.covars],
        infer_hidden=False,
    )
    f.suptitle("Expectation-Maximization Solution, Feature 0", size=16)

    f = helpers.gaussian_hinton_diagram(
        model.pi,
        model.A,
        model.means[:,1].ravel(),
        [x[1][1] for x in model.covars],
        infer_hidden=False,
    )
    f.suptitle("Expectation-Maximization Solution, Feature 1", size=16)

    for i in range(0, num_features - 1):
        for j in range(i+1, num_features):
            means, covars = helpers.extract_paired_dist(model.means, model.covars, i, j)
            f = helpers.multi_gaussian_3d_plot(means, covars, xlabel="Feature " + str(i), ylabel="Feature " + str(j))

    plt.show()
