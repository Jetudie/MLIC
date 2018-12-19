import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math

def prior_uniform(n):
    prior = []
    for _ in range(n):
        prior.append(1/n)
    return prior

def combination(xy):
    c = 1
    for n in range(min(xy)):
        c *= xy[0]+xy[1]-n
    c /= math.factorial(min(xy))    
    return c

def likelihood_binomial(observation, value):
    likelihood = []
    MLE = []
    # Likelihood
    c = combination(observation)
    for n in range(len(value)):
        # likeliyhood = binomial distribution formula used on observation
        likelihood.append(c*math.pow(value[n], observation[0])*math.pow(1-value[n], observation[1]))
    
    # MLE
    MLE = []
    n = observation[0]
    for _ in range(2*n-1):
        MLE.append(n)
    return likelihood

def plot(x_axis, y_axis, name):
    y_pos = np.arange(len(x_axis))
    plt.bar(y_pos, y_axis)
    plt.xticks(y_pos, x_axis)
    plt.title(name)
    plt.savefig(name)
    plt.show()

def toss(value, prior, observation):
    likelihood = likelihood_binomial(observation, value)
    posterior = []
    for i in range(len(value)):
        posterior.append(likelihood[i]*prior[i])
    s = sum(posterior)
    for i in range(len(posterior)):
        posterior[i] /= s
    plot(value, prior, "Prior")
    plot(value, likelihood, "Likelihood")
    plot(value, posterior, "Posterior")
    # max = sorted(posterior)[len(posterior)-1]
    max_posterior = max(posterior)
    m = posterior.index(max_posterior)
    print("MAP value(p):", value[m], ", posterior:", max_posterior)

if __name__ == "__main__":
    # let observation[0]: head, observation[1]: tail
    Observation = [2, 8]
    # possible prior value (probability of head: p)
    Values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, \
            0.6, 0.7, 0.8, 0.9, 1.0]
    N = len(Values)
    # probability distribution of prior value
    Priors_a = prior_uniform(N)
    Priors_b = [0.01, 0.01, 0.05, 0.08, 0.15, \
            0.4, 0.15, 0.08, 0.05, 0.01, 0.01]
    toss(Values, Priors_a, Observation)
    toss(Values, Priors_b, Observation)