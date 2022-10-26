import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

# Dati alfa in minute
alf = 4.8
alf = alf / 60

if __name__ == '__main__':
    model = pm.Model()
    with model:
        clienti = pm.Poisson('CL', 20)
        plata = pm.TruncatedNormal('P', 0.017, 0.0083, lower=0)
        cook = pm.Exponential('CO', 1/alf)
        # Serve este timpul necesar ca un client sa fie servit, de cand intra in magazin si pana iese
        serve = pm.Deterministic('S', plata + cook)
        trace = pm.sample(20000)

    dictionary = {
                'clienti': trace['CL'].tolist(),
                'plata': trace['P'].tolist(),
                'cook': trace['CO'].tolist(),
                'serve': trace['S'].tolist()
                }
    df = pd.DataFrame(dictionary)

    az.plot_posterior(trace)
    plt.show()

    # Pentru punctul b) am incercat mai multe valori ale lui alfa si am ajuns la concluzia ca alfa = 4,8 este aproximativ valoarea cautata
    # Valoarea superioara a intervalului HDI pentru acesta e 0.25 ore, adica 15 min (pe graficul lui "serve")
    # Media lui serve bazata pe alfa este afisata tot pe grafic si e aprox. 0.1 ore, adica 6 min