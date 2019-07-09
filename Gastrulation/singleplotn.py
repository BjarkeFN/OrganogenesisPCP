import numpy as np
from mayavi import mlab
import pickle
import scipy.io
import glob
import os
import sys
print(sys.argv)
k=sys.argv[1]
#mlab.options.backend = 'envisage'
mlab.figure(figure=None, bgcolor=None, fgcolor=None, engine=None, size=(1000, 800))
#list_of_files = glob.glob('./mats_8_neuru/*.mat') # * means all if need specific format then *.csv
list_of_files = glob.glob(f'./mats_{k}/*.mat') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)


#fname = 'data/sim_2400_normal_q.npy'

#P = scipy.io.loadmat('mats/t10.mat');
if len(sys.argv) == 3:
    n=sys.argv[2]
    P = scipy.io.loadmat(f'./mats_{k}/t{n}.mat');
else:
    P = scipy.io.loadmat(latest_file);
    print(latest_file)

#P = P['p']
x = P['x']
p = P['p']
q = P['q']
a = P['alpha']
a = a[:,0]
#pdiv = P['pdiv']
#print("Pshape", P.shape)
#x=P[:,0:3]
#p=P[:,3:6]
#q=P[:,6:9]
print("xshape", x.shape)
print("pshape", p.shape)
print("qshape", q.shape)
#with open(fname, 'rb') as f:
#    x, p, q, a = pickle.load(f)

#t = -1

#print(len(x[t]))

# plot = mlab.points3d(x[t][:, 0], x[t][:, 1], x[t][:, 2], scale_factor=1, scale_mode='none')

# colors = a[t] - np.mean(a[t])
# colors /= 2 * np.max(colors)
# colors += 0.5

# plot.mlab_source.dataset.point_data.scalars = colors

# Normalize p,q

q /= np.sqrt(np.sum(q ** 2, axis=1))[:, None]

colors = abs(p[:,1]**2+p[:,0]**2)**0.2

# Positional color modulation:
xa = abs(x[:,1])/max(abs(x[:,1]))
print(xa)
colors = colors*(1-xa)**3

print("Colshape:", colors.shape)
# Alpha modulation:
colors = colors+a
print("Alphashape:", a.shape)
print("Colshape:", colors.shape)

colors = colors - np.mean(colors)

# Let's see if we can restrict the range of 'colors'
colors -= min(colors)
colorrange = max(colors)-min(colors)

colors /= colorrange
colors *= 0.2

# Exchange apical and basal:
colors = 1-colors

print("Colormax:", max(colors))
print("Colormin:", min(colors))

# color by direction of PCP in XY plane
#colors = abs(p[:,1]**2+p[:,0]**2)**0.2
#colors = np.arctan2(q[:,1],q[:,0])/(2*np.pi)
#colors = colors - np.mean(colors)
#colors = abs(x[:,0]**2)**0.2
#colors = colors - np.mean(colors)

#colors = abs((q[:,0]**2)**0.2)
#colors = colors - np.mean(colors)



#p = q
x0 = 0
y0 = 10
idx = x[:, 1] < y0
#idx = x[:, 1] > 0

dp = 0.1
xAB = x + dp*p

scalefactor = 2.4
#scalefactor = 0

xArrow = x + 2*p

colors = colors[idx]

PlotAB = True

ballres=10
ballmode='sphere' # The mode of the glyphs.
# Must be ‘2darrow’ or ‘2dcircle’ or ‘2dcross’ or ‘2ddash’ or ‘2ddiamond’ or ‘2dhooked_arrow’ or ‘2dsquare’ or ‘2dthick_arrow’
# or ‘2dthick_cross’ or ‘2dtriangle’ or ‘2dvertex’ or ‘arrow’ or ‘axes’ or ‘cone’ or ‘cube’ or ‘cylinder’ or ‘point’ or ‘sphere’. Default: sphere

#plot = mlab.points3d(x[:, 0], x[:, 1], x[:, 2], scale_factor=2, scale_mode='none')
plot = mlab.points3d(x[idx, 0], x[idx, 1], x[idx, 2], scale_factor=scalefactor, scale_mode='none', resolution=ballres, mode=ballmode)

if PlotAB:
    plotAB = mlab.points3d(xAB[idx, 0], xAB[idx, 1], xAB[idx, 2], scale_factor=scalefactor, scale_mode='none', resolution=ballres, mode=ballmode)
    plotAB.mlab_source.dataset.point_data.scalars = 1-colors

#colors = a - np.mean(a)
#colors = pdiv - np.mean(pdiv)
#colors = (a>0) - np.mean(a>0) + (pdiv>0) - np.mean(pdiv>0)
#colors /= 2 * np.max(colors)
#colors += 0.5

plot.mlab_source.dataset.point_data.scalars = colors


#ABPs = mlab.quiver3d(x[:, 0], x[:, 1], x[:, 2], p[:, 0], p[:, 1], p[:, 2], scalars=colors, mode="sphere", scale_factor=2*scalefactor)

#mlab.quiver3d(x[t][idx, 0], x[t][idx, 1], x[t][idx, 2], p[t][idx, 0], p[t][idx, 1], p[t][idx, 2])

#mlab.quiver3d(xArrow[idx, 0], xArrow[idx, 1], xArrow[idx, 2], p[idx, 0], p[idx, 1], p[idx, 2], scale_factor=scalefactor, mode='arrow')
#mlab.quiver3d(x[idx, 0], x[idx, 1], x[idx, 2], p[idx, 0], p[idx, 1], p[idx, 2], scale_factor=0.75*scalefactor, mode='cylinder')
#mlab.quiver3d(xArrow[idx, 0], xArrow[idx, 1], x[idx, 2], q[idx, 0], q[idx, 1], q[idx, 2], scale_factor=3, scale_mode='none', mode='cylinder')
#mlab.view(azimuth=92, elevation=67, distance=150, focalpoint=(0,0,0))
mlab.view(azimuth=60, elevation=90, distance=160, focalpoint=(0,0,0))
mlab.show()
#mlab.view(azimuth=92, elevation=67, distance=202, focalpoint=(0,0,0))
#mlab.savefig(f'pngs/blap{str(n).zfill(4)}.png')
