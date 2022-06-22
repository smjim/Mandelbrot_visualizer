import numpy as np
import pylab as plt

data = np.loadtxt("data.txt", dtype=int)

fig, ax2 = plt.subplots()
ax2.imshow(data, extent=[-2.00,0.47,-1.12,1.12])

plt.show()
