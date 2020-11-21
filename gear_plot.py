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


def gear(teeth, center=(0, 0), rot: float = 0, module: float = 1):
	"""
	Plot a gear
	:param teeth:	Number of teeth in gear
	:param center:  Center of gear
	:param rot: 	Rotation in #teeth
	:param module:	Module of gear
	"""
	pressure_angle = 20
	addendum = module
	dedendum = module*1.25
	pitch_radius = module*teeth/2
	pitch = module*pi
	rot *= pitch
	tooth = pitch/2
	addendum_offset = addendum*tan(radians(pressure_angle))
	dedendum_offset = dedendum*tan(radians(pressure_angle))
	print(pitch, tooth, addendum_offset)
	#teeth = 2
	plot(circle(pitch_radius, c=center), color='yellow')
	plot(circle(pitch_radius+addendum, c=center), color='yellow')
	plot(circle(pitch_radius-dedendum, c=center), color='yellow')
	plot(circle(pitch_radius-addendum, c=center), color='cyan')
	plot(circle(2, c=center), color='red')
	plot(circle(1, c=center), color='blue')

	tooth_top_hw = tooth/2 - addendum_offset
	for n in range(teeth):
		plot(involute(pitch_radius-addendum, a=(n*pitch+rot+tooth_top_hw)/pitch_radius, c=center), 'red')
		plot(involute(pitch_radius-addendum, a=(n*pitch+rot-tooth_top_hw)/pitch_radius, c=center, up=-1), 'blue')


# print(circle(2))
# plot(circle(1, (1, -.5)), color='blue')plt.grid()
gear(19, rot=0.25)
# gear(30)
gear(15, (34/2, 0), rot=-0.25)
plt.axis('equal')
# Set zoom_radius to zoom in around where gears meet
zoom_radius = 2
if zoom_radius:
	ax = plt.gca()
	ax.set_xlim(10 - zoom_radius, 10 + zoom_radius)
	ax.set_ylim(-zoom_radius, zoom_radius)
plt.show()

