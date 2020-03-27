import numpy as np
import matplotlib.pyplot as plt
import CubicFields as cf


j = 8
W = 1
xvals = np.arange(-1, 1, 1e-3)
evals = [np.sort(cf.energies(j, W, x)) for x in xvals]
plt.figure()
plt.plot(xvals, evals)
plt.show()
