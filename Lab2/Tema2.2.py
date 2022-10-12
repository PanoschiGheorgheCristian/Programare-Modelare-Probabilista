import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

S1 = stats.gamma.rvs(a = 4, scale = 1/3, size = 10000)
S2 = stats.gamma.rvs(a = 4, scale = 1/2, size = 10000)
S3 = stats.gamma.rvs(a = 5, scale = 1/2, size = 10000)
S4 = stats.gamma.rvs(a = 5, scale = 1/3, size = 10000)
latenta = stats.expon.rvs(scale = 4, loc = 0, size = 10000)
X = []

# X = stats.bernoulli.rvs(size = 10000, p = 0.4) * M1 + stats.bernoulli.rvs(size = 10000, p = 0.6) * M2

for i in range(10000):
    random_nr = np.random.rand()
    if random_nr <= 0.25:
        X.append(S1[i] + latenta[i])
    elif random_nr <= 0.5:
        X.append(S2[i] + latenta[i])
    elif random_nr <= 0.8:
        X.append(S3[i] + latenta[i])
    else :
        X.append(S4[i] + latenta[i])

x = np.array(X)

count = 0

for i in range(1,10000):
    if X[i] > 3:
        count = count + 1

print(count/10000)

plt.subplot(1,5,1)
plt.plot(S1)

plt.subplot(1,5,2)
plt.plot(S2)

plt.subplot(1,5,3)
plt.plot(S3)

plt.subplot(1,5,4)
plt.plot(S4)

plt.subplot(1,5,5)
plt.plot(latenta)

az.plot_posterior(x)
plt.show()
