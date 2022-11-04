import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == '__main__':
    data = np.array(pd.read_csv("data.csv"))

    X = data[:, 3]
    Y = data[:, 1]

    #Reprezentare grafica a datelor:
    plt.subplot(1,2,1)
    plt.plot(X)

    plt.subplot(1,2,2)
    plt.plot(Y)
    plt.show()

    # Model de regresie liniara:
    basic_model = pm.Model()

    with basic_model:
        
        alpha = pm.Normal("alpha", mu=0, sigma=10, shape=400)
        beta = pm.Normal("beta", mu=0, sigma=10, shape=400)
        eps = pm.HalfCauchy('eps', 5, shape=400)
        
        # Expected value of outcome
        mu = pm.Deterministic('mu', alpha + beta * X + eps)

        Y_obs = pm.Normal("Y_obs", mu=mu, sd=eps, observed=Y)
        idata_g = pm.sample(20, tune=20, return_inferencedata=True)

    #Fit model to data and find MAP estimates
    map_estimate = pm.find_MAP(model=basic_model)
    print(map_estimate)

#A ramas sa aflu concludent care e raspunsul la 3. din MAP