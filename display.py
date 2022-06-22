import numpy as np
import pylab as plt

data = np.loadtxt("data.txt", dtype=int)

plt.imshow(data, cmap = 'plasma', origin = 'lower',  extent = [-2.00,0.47,-1.12,1.12])
plt.xlabel('Real')
plt.ylabel('Imaginary')

plt.colorbar();
plt.show()
