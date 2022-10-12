import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

M1 = stats.expon.rvs(scale = 1/4, loc = 0, size = 10000)
M2 = stats.expon.rvs(scale = 1/6, loc = 0, size = 10000)
X = []

# X = stats.bernoulli.rvs(size = 10000, p = 0.4) * M1 + stats.bernoulli.rvs(size = 10000, p = 0.6) * M2

for i in range(10000):
    random_nr = np.random.rand()
    if random_nr <= 0.4 :
        X.append(M1[i])
    else:
        X.append(M2[i])

x = np.array(X)

plt.subplot(1,2,1)
plt.plot(M1)

plt.subplot(1,2,2)
plt.plot(M2)

az.plot_posterior(x)
print(np.std(X))
plt.show()