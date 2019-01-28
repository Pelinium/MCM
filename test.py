import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

a =np.array([[156, 138, 156],
             [1300, 137, 156],
             [138, 138, 1300],
             [137, 137, 137]])

ca = np.array([[156,200,200,200],
               [138,170,255,245],
               [137,208,130,40],
               [1300,63,165,76]])

u, ind = np.unique(a, return_inverse=True)
b = ind.reshape((a.shape))

colors = ca[ca[:,0].argsort()][:,1:]/255.
cmap = matplotlib.colors.ListedColormap(colors)
norm = matplotlib.colors.BoundaryNorm(np.arange(len(ca)+1)-0.5, len(ca))

plt.imshow(b, cmap=cmap, norm=norm)

cb = plt.colorbar(ticks=np.arange(len(ca)))
cb.ax.set_yticklabels(np.unique(ca[:,0]))

plt.show()