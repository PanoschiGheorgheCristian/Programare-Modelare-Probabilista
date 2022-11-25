import matplotlib as plt
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import scipy as sp

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

if __name__ == '__main__':
    data = np.array(pd.read_csv("Admission.csv"))
    
    admitted = data[:, 0]
    GRE = data[:, 1]
    GPA = data[:, 2]
    
    admitted_p_true = np.count_nonzero(admitted) / admitted.shape[0]
    admitted_distribution = sp.stats.binom.rvs( n=10000, p=admitted_p_true, size=admitted.shape[0])
    
    admission = pm.Model()
    
    with admission:
        beta0 = pm.Normal('beta0', mu=0, sd=2)
        beta1 = pm.Normal('beta1', mu=0.5, sd=1)
        beta2 = pm.Normal('beta2', mu=1, sd=1)

        p = pm.Deterministic('p', sigmoid(beta0 + beta1 * GRE + beta2 * GPA))
    

        y_obs = pm.Binomial('y_obs', n=10000, p=p, observed=admitted_distribution)

        trace = pm.sample(1000, tune=100)
    
    # p_out = trace['p'].tolist()
    # print("////////")
    # print(min(p_out))
    # print(max(p_out))
    # print("////////")
    
    ppc = pm.sample_posterior_predictive(trace, samples=2000, model=admission)
    sig = az.plot_hdi(GPA, GRE, ppc['y_obs'], hdi_prob=0.97, color='k')
    plt.xlabel('GPA & GRE')
    plt.ylabel('Admitted', rotation=0)