import matplotlib.pyplot as plt
import arviz as az
import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == '__main__':
    data = np.array(pd.read_csv("Prices.csv"))
    
    prices = data[:, 0]
    freq = data[:, 1]
    hard_drive = data[:, 2]
    for i in range(hard_drive.size):
        hard_drive[i] = np.log(hard_drive[i])
    # ram = data[:, 3]
    # premium = data[:, 4]
    
    computer_prices = pm.Model()
    
    with computer_prices:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0.5, sd=5)
        beta2 = pm.Normal('beta2', mu=1, sd=15)
        sigma = pm.HalfNormal('sigma', sd=1)
        
        mu = pm.Deterministic('mu', alpha + beta1 * freq + beta2 * hard_drive)
        
        y = pm.Normal('y', mu=mu, sd=sigma, observed=prices)
        
        trace = pm.sample(2000, tune=2000, cores=4)
        
    ppc = pm.sample_posterior_predictive(trace, samples=2000, model=computer_prices)
    sig = az.plot_hdi(freq, hard_drive, ppc['y'], hdi_prob=0.97, color='k')
    plt.xlabel('Freq & hard_drive')
    plt.ylabel('Price', rotation=0)
    
    
    