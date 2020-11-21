import matplotlib.pyplot as plt
from math import *


def circle(r, c=(0, 0)):
	cx, cy = c
	pt = lambda t: (r*sin(t)+cx, r*cos(t)+cy)
	steps = 360
	return [pt(step/steps*tau) for step in range(steps+1)]


def plot(xy, color='black'):
	plt.plot(*zip(*xy), color)


def involute(r, a=0, up=1, c=(0, 0)):
	# print('inv: r=%3.1f a=%8.5f up=%d' % (r, a, up))
	# See https://en.wikipedia.org/wiki/Involute#Involutes_of_a_circle
	# Use arc length as approximation for height.
	# It will always be a shorter, though.
	#             al = r/2 * sqr(t)
	#  therefore, t = sqrt(al/r*2)
	al = 2.25		# addendum+dedendum
	tt = sqrt(al/r*2)
	cx, cy = c
	pt = lambda t: (r*(cos(t)+(t-a)*sin(t))+cx, r*(sin(t)-(t-a)*cos(t))+cy)
	steps = 60
	return [pt(up*step/steps*tt+a) for step in range(steps+1)]


def gear(teeth, center=(0,0)):
	module = 1
	pa = 20
	ad = module
	dd = module*1.25
	pr = module*teeth/2
	pitch = module*pi
	tooth = pitch/2
	ado = ad*tan(radians(pa))
	ddo = dd*tan(radians(pa))
	print(pitch, tooth, ado)
	#teeth = 2
	plot(circle(pr, c=center), color='yellow')
	plot(circle(pr+ad, c=center), color='yellow')
	plot(circle(pr-dd, c=center), color='yellow')
	plot(circle(pr-ad, c=center), color='cyan')
	plot(circle(2, c=center), color='red')
	plot(circle(1, c=center), color='blue')

	tooth_top_hw = tooth/2 - ado
	for n in range(teeth):
		plot(involute(pr-ad, a=(n*pitch+tooth_top_hw)/pr, c=center), 'red')
		plot(involute(pr-ad, a=(n*pitch-tooth_top_hw)/pr, c=center, up=-1), 'blue')

# print(circle(2))
# plot(circle(1, (1, -.5)), color='blue')plt.grid()
gear(19)
# gear(30)
gear(15, (34/2, 0))
plt.axis('equal')
# Set zoom_radius to zoom in around where gears meet
zoom_radius = 2
if zoom_radius:
	ax = plt.gca()
	ax.set_xlim(10 - zoom_radius, 10 + zoom_radius)
	ax.set_ylim(-zoom_radius, zoom_radius)
plt.show()

