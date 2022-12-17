import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

if __name__ == "__main__":
    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, 3]
    std_devs = [2, 2, 2]
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    az.plot_kde(np.array(mix))

    plt.show()

    clusters = 2
    with pm.Model() as model_2:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        idata_mg = pm.sample(random_seed=123, return_inferencedata=True)

    clusters = 3
    with pm.Model() as model_3:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        idata_mg = pm.sample(random_seed=123, return_inferencedata=True)

    clusters = 4
    with pm.Model() as model_4:
        p = pm.Dirichlet('p', a=np.ones(clusters))
        means = pm.Normal('means', mu=mix.mean(), sd=10, shape=clusters)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)
        idata_mg = pm.sample(random_seed=123, return_inferencedata=True)