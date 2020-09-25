from math import *

def bevel_calc(teeth1, teeth2, module=1., face_width=5., angle=90.):
    """Calculate the height for gear teeth1 for bevel gears.
       This is based on the diametrical pitch being at the base.
    """

    radius1 = module * teeth1 / 2.
    radius2 = module * teeth2 / 2.
    alpha = radians(angle)
    beta = radians(90. - angle)

    width2 = radius1 + radius2 * cos(alpha)
    height1 = radius2 * sin(alpha)
    height2 = sin(beta) * width2 / sin(alpha)
    height = height1 + height2

    hypot = sqrt(height * height + radius1 * radius1)
    bevel_angle = asin(height / hypot)

    addendum = module
    dedendum = module * 1.25

    blank_thickness = dedendum * cos(bevel_angle) +\
                      face_width * sin(bevel_angle) +\
                      addendum * cos(bevel_angle) * ((height - face_width * sin(bevel_angle)) / height)

    # Distance from face of blank to reference pitch circle
    blank_face_offset = blank_thickness - dedendum * cos(bevel_angle)

    # Minimum diameter for the blank to work
    blank_diameter = teeth1 * module + 2 * addendum * sin(bevel_angle)

    return height, bevel_angle, blank_thickness, blank_diameter, blank_face_offset

import matplotlib.pyplot as plt
import numpy

teeth1 = 15
teeth2 = 15
module = 1.
face_width = 2.5
angle = 90.
addendum = module
dedendum = module * 1.25

height, bevel_angle, blank_thickness, blank_diameter, blank_face_offset = bevel_calc(teeth1, teeth2, module, face_width, angle)

blank_radius = blank_diameter / 2.
pitch_radius = module * teeth1 / 2.
fig = plt.figure()

# Blank
xy = [(-blank_radius, -height+blank_face_offset), (blank_radius, -height+blank_face_offset),
      (blank_radius, -height+blank_face_offset-blank_thickness), (-blank_radius, -height+blank_face_offset-blank_thickness),
      (-blank_radius, -height+blank_face_offset)]
plt.plot(*numpy.array(xy).swapaxes(0,1), color='g')

# Pitch diameter
xy = [(-pitch_radius, -height), (pitch_radius, -height)]
plt.plot(*numpy.array(xy).swapaxes(0,1), color='y')

# Tooth
xy = [(-pitch_radius, -height), (0,0),
      (-pitch_radius - addendum * sin(bevel_angle), -height + addendum * cos(bevel_angle)),
      (-pitch_radius + dedendum * sin(bevel_angle), -height - dedendum * cos(bevel_angle)), (0,0)]
plt.plot(*numpy.array(xy).swapaxes(0,1), 'b--')

# Tooth face width
xy = [(-pitch_radius, -height), (-pitch_radius + face_width * cos(bevel_angle) , -height + face_width * sin(bevel_angle))]
plt.plot(*numpy.array(xy).swapaxes(0,1), 'r')


plt.axis('equal')
plt.savefig('k.png')
