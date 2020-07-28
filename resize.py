#!/usr/bin/python3

from math import *
from gcode import *

maxOuterRadius  = 4.53 / 2.
desiredRadius   = 4.443 / 2.
teeth           = 111
metric          = False     # True for metric (mm), False for imperial (in)
thickness       = .063

toolNumber      = 2
spindleSpeed    = 8000
feedRate        = 15.
axisFeedRate    = 500.
cutterDiameter  = .25

# Extra cutting parameters
depthOfCut      = .005
safeDistance    = .1
leadInOut       = .2


# Calculate internal working constants
units           = 'mms' if metric else 'inches'
cutterRadius    = cutterDiameter / 2.
anglePerTooth   = 360./teeth


# Preamble of parameter comments to assist machine setup
g = Gcode()
g.append( '%' )
g.comment( 'Spur gear blank trimming' )
g.comment()
g.comment( "Max Outer Radius: %g %s" % (maxOuterRadius,units))
g.comment( "Desired Radius: %g %s" % (desiredRadius,units))
g.comment( "Teeth: %d" % teeth )
g.comment( "Thickness: %g %s" % (thickness,units))
g.comment()

# Setup the machine, choose the tool, set the rates
g.comment( 'T%d D=%g WOC=%g - End mill' % (toolNumber, cutterDiameter, depthOfCut))
g.append( 'G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49' )
g.append( 'G21 (mm)' if metric else 'G20 (inch)' )
g.append( 'G30' )
g.append( 'T%d G43 H%d M6' % (toolNumber, toolNumber))
g.append( 'S%d M3 M8' % spindleSpeed )
g.append( 'G54' )
g.append( 'F%g' % feedRate )

# depths are the various passes for trimming the blank
depths = []
depth = maxOuterRadius - desiredRadius
while depth > 0:
    depth = max( depth - depthOfCut, 0.0 )
    depths.append(depth)

# Generate the gcode for actually trimming the blank
g.move(x=thickness+cutterRadius+safeDistance)
g.move(y=0)
g.move(a=0.)
for depth in depths:
    g.move(z=desiredRadius+depth)
    g.cut(x=0.0)
    g.append( 'G1 A360. F%g' % axisFeedRate )
    g.move(x=thickness+cutterRadius+safeDistance)
    g.move(a=0.)

# Program is done, shutdown time
g.append( 'M05 M09' )
g.append( 'G30' )
g.append( 'M30' )
g.append( '%' )

print(g.output())
