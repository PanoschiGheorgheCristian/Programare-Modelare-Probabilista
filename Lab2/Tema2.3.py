import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

SS=[]
SB=[]
BS=[]
BB=[]
for j in range(1,100):
    nr_ss = 0
    nr_sb = 0
    nr_bs = 0
    nr_bb = 0
    for i in range(1,10):
        random_nr = np.random.rand()
        if random_nr < 0.15:
            nr_ss = nr_ss + 1
        elif random_nr < 0.5:
            nr_sb = nr_sb + 1
        elif random_nr < 0.65:
            nr_bs = nr_bs + 1
        else:
            nr_bb = nr_bb + 1
    SS.append(nr_ss)
    SB.append(nr_sb)
    BS.append(nr_bs)
    BB.append(nr_bb)


ss = np.array(SS)
sb = np.array(SB)
bs = np.array(BS)
bb = np.array(BB)

az.plot_posterior({'ss':ss, 'sb':sb, 'bs':bs, 'bb':bb})

plt.show()