#!/usr/bin/python3

import sys
from math import radians, degrees, pi, sqrt
import configargparse
from gcode import *


p = configargparse.ArgParser(
    default_config_files=['resize.cfg'],
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    description="Generate G Code to trim a gear blank to size.")

p.add('out', nargs='?', type=configargparse.FileType('w'), default=sys.stdout)
p.add('--config', '-X', is_config_file=True, help='Config file path')

# Tool arguments
p.add('--diameter', '-I', type=float, default=15., help='Tool: cutting diameter at tip in mm')
p.add('--number', '-N', type=int, default=1, help='Tool: tool number')
p.add('--rpm', '-R', type=float, default=2000., help='Tool: spindle speed')
p.add('--feed', '-F', type=float, default=200., help='Tool: feed rate')
p.add('--mist', '-M', action='store_true', help='Tool: turn on mist coolant')
p.add('--flood', '-L', action='store_true', help='Tool: turn on flood coolant')
p.add('--mill', default='conventional', choices=['both', 'climb', 'conventional'], help='Tool: cutting method')

# Blank arguments
p.add('--thick', '-k', type=float, required=True, help='Thickness of gear blank in mm')
p.add('--rough', '-g', type=float, required=True, help='Rough diameter of the blank in mm')
p.add('--finish', '-f', type=float, required=True, help='Finished diameter for the blank in mm')
p.add('--angle', '-a', type=float, default=10., help='Rotation per clearing operation in degrees')
p.add('--steps', '-p', type=int, default=1, help='Number of equidistant passes by the cutter per operation')
p.add('--right', '-r', action='store_true', help='Rotary axis is on the right side of the machine')
p.add('--clear', '-c', type=float, default=2., help='Cutter clearance from gear blank in mm')

args = p.parse_args()
out = args.out

tool_diameter = args.diameter
tool_radius = tool_diameter / 2.
tool_number = args.number
tool_rpm = args.rpm
tool_feed = args.feed
cutter_clearance = args.clear
outer_diameter = args.rough
outer_radius = outer_diameter / 2.
inner_diameter = args.finish
inner_radius = inner_diameter / 2.
blank_thickness = args.thick
turn_angle = radians(args.angle)
angle_direction = 1 if args.right else -1
mill = args.mill

steps = args.steps
out = args.out

# Preamble of parameter comments to assist machine setup
g = Gcode()
g.append('%')
g.comment('Spur gear blank trimming')
g.comment()
g.comment("Rough Diameter: %g mm" % outer_diameter)
g.comment("Finish Diameter: %g mm" % inner_diameter)
g.comment("Thickness: %g mm" % blank_thickness)
g.comment()

# Setup the machine, choose the tool, set the rates
#g.comment('T%d D=%g WOC=%g - End mill' % (toolNumber, cutterDiameter, depthOfCut))
g.append('G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49')
g.append('G21')
g.append('G30')
g.append('T{number} G43 H{number} M6'.format(number=tool_number))
g.append('S%d M3 M8' % tool_rpm)
g.append('G54')
g.append('F%g' % tool_feed)

# depths are the various passes for trimming the blank
depths = []
cut_step = (outer_radius - inner_radius) / steps
x_offset = cutter_clearance + blank_thickness / 2.
x_start, x_end = -angle_direction * x_offset, angle_direction * x_offset
cut_radius = outer_radius - cut_step

x = x_end if mill == 'climb' else x_start
g.move(x=x)
x = -x

# Cut the blank to size
while cut_radius >= inner_radius:
    z = -sqrt(cut_radius**2 - (cut_radius-cut_step)**2)
    g.move(z=z)
    angle = 0.
    y = cut_radius + tool_radius
    g.move(y=y)
    while angle < 2 * pi:
        g.move(a=angle_direction * degrees(angle))
        g.cut(x=x)
        if mill == 'conventional':
            g.move(y=y+cutter_clearance)
            g.move(x=-x)
            g.move(y=y)
        elif mill == 'climb':
            g.move(y=y+cutter_clearance)
            g.move(x=-x)
            g.move(y=y)
        else:
            x = -x
        angle += turn_angle
    cut_radius -= cut_step

# Program is done, shutdown time
g.append('M05 M09')
g.append('G30')
g.append('M30')
g.append('%')

print(g.output(), file=out)
