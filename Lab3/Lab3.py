import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


if __name__ == '__main__':
    model = pm.Model()
    with model:
        cutremur = pm.Bernoulli('C', 0.0005)
        incendiu_p = pm.Deterministic('I_p', pm.math.switch(cutremur, 0.03, 0.01))
        incendiu = pm.Bernoulli('I', p=incendiu_p)
        alarma_p = pm.Deterministic('A_p', pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.02), pm.math.switch(incendiu, 99.9999, 0.0001)))
        alarma = pm.Bernoulli('A', p=alarma_p, observed = 1)
        trace = pm.sample(20000)

    dictionary = {
                'cutremur': trace['C'].tolist(),
                'incendiu': trace['I'].tolist()
                }
    df = pd.DataFrame(dictionary)

    p_cutremur_alarma = df[ (df['cutremur'] == 1) ].shape[0] / df.shape[0]

    print(p_cutremur_alarma)

    az.plot_posterior(trace)
    plt.show()