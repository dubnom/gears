import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from math import *


fig, ax = plt.subplots()

Path = mpath.Path

def circle(r, nPoints=30):
    aa = 2. * pi / nPoints
    return [(r*cos(a*aa), r*sin(a*aa)) for a in range(nPoints+1)]

outerCircle = circle(1.)

path_data = [
    (Path.MOVETO, (1.58, -2.57)),
    (Path.CURVE4, (0.35, -1.1)),
    (Path.CURVE4, (-1.75, 2.0)),
    (Path.CURVE4, (0.375, 2.0)),
    (Path.LINETO, (0.85, 1.15)),
    (Path.CURVE4, (2.2, 3.2)),
    (Path.CURVE4, (3, 0.05)),
    (Path.CURVE4, (2.0, -0.5)),
    (Path.CLOSEPOLY, (1.58, -2.57)),
    ]
codes, verts = zip(*path_data)
path = mpath.Path(verts, codes)
patch = mpatches.PathPatch(path, facecolor='r', alpha=0.5)
ax.add_patch(patch)

# plot control points and connecting lines
x, y = zip(*path.vertices)
line, = ax.plot(x, y, 'go-')

x, y = zip(*outerCircle)
line, = ax.plot(x, y, 'b-')

ax.grid()
ax.axis('equal')
plt.savefig('foobar.png')

