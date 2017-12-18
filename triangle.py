"""
    triangle.py -   Generate gcode for spur, crown, (and coming soon - bevel) gears using a triangular
                    tooth profile.

                    Orignally written by Michael Dubno and for general purpose use - no copyrights.
"""
from math import *
from gcode import *

# Gear description
teeth           = 48
metric          = False     # True for metric (mm), False for imperial (in)
module          = .5
thickness       = .0625     # Thickness of the gear surface (or edge of a crown)
gearType        = 'Crown'   # Choices are: Spur, Crown, or Bevel

# Cutting tool description
toolNumber      = 10
spindleSpeed    = 3000
feedRate        = 24.
cutterDiameter  = .75
cutterAngle     = 60.
centerlineOffset= .100      # distance from bottom of tool to edge cutting point

# Extra cutting parameters
depthOfCut      = .04       # Maximum depth of cut
safeDistance    = .1        # Distance to retract from gear blank
leadInOut       = .2        # Extra distance added to the entry and exit of the cut


# Calculate internal working variables
module          = module if metric else module / 25.4
units           = 'mms' if metric else 'inches'
diameter        = teeth * module
radius          = diameter / 2.
circumference   = diameter * pi
toothHeight     = sin(radians(cutterAngle)) * circumference/teeth
addendum        = toothHeight / 2.
dedendum        = toothHeight / 2.
innerRadius     = radius-dedendum
extraAngle      = 180./teeth
cutterRadius    = cutterDiameter / 2.
anglePerTooth   = 360./teeth


# Angle passes are offsets to a centerline cutting setup to allow a standard fixed size cutter to create
# included angles larger than the cutter's natural angle.  This is done by slightly rotating the gear up and down
# while adjusting the height and y distance from the blank.
anglePasses = [
        # aOffset,    rOffset,                                            zOffset
        (-extraAngle, innerRadius - cos(radians(extraAngle))*innerRadius, -sin(radians(extraAngle))*innerRadius),
        (extraAngle,  innerRadius - cos(radians(extraAngle))*innerRadius, sin(radians(extraAngle))*innerRadius)
]

# depthPasses are the depths that the cutter should use for cutting the teeth - sometimes multiple passes are needed due
# to the rigidity of the cutter and materials
depthPasses = []
depth = 0.
while depth < toothHeight:
    depth = min(depth+depthOfCut, toothHeight)
    depthPasses.append(depth)

# Preamble of parameter comments to assist machine setup
g = Gcode()
g.append( '%' )
g.comment( "%s Triangle gear cutting" % gearType )
g.comment( "Teeth: %d,  Module: %g %s,  Thickness: %g %s" % (teeth, module, units, thickness, units))
g.comment()
g.comment( "Diameter: %g %s" % (diameter, units))
g.comment( "Radius: %g %s" % (radius, units))
g.comment( "Circumference: %g %s" % (circumference, units))
g.comment( "ToothHeight: %g %s" % (toothHeight, units))
g.comment( "Gear Blank Diameter: %g %s" % (diameter + 2.*addendum, units))
g.comment( "ExtraAngle: %g degrees" % extraAngle )
g.comment()

# Setup the machine, choose the tool, set the rates
g.comment( 'T%d D=%g - Dual bevel %g degree cutter' % (toolNumber, cutterDiameter, cutterAngle))
g.append( 'G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49' )
g.append( 'G21 (mm)' if metric else 'G20 (inch)' )
g.append( 'G30' )
g.append( 'T%d G43 H%d M6' % (toolNumber, toolNumber))
g.append( 'S%d M3 M8' % spindleSpeed )
g.append( 'G54' )
g.append( 'F%g' % feedRate )

# Finally generate the gear teeth
if gearType == 'Spur':
    g.move(y=radius+cutterRadius+safeDistance)
    g.move(0.,leadInOut,z=0.)
    for tooth in range(teeth):
        g.comment( 'Tooth %d' % tooth )
        for aOffset,rOffset,zOffset in anglePasses:
            g.move(tooth*anglePerTooth+aOffset,z=zOffset-centerlineOffset)
            for depth in depthPasses:
                g.move(y=radius+cutterRadius+addendum-rOffset-depth)
                g.cut(x=-thickness-leadInOut)
                g.move(y=radius+cutterRadius+addendum+safeDistance)
                g.move(x=leadInOut)

elif gearType == 'Crown':
    g.move(x=cutterRadius+safeDistance)
    g.move(0.,y=radius,z=0.)
    for tooth in range(teeth):
        g.comment( 'Tooth %d' % tooth )
        for aOffset,rOffset,zOffset in anglePasses:
            g.move(tooth*anglePerTooth+aOffset,z=zOffset-centerlineOffset)
            for depth in depthPasses:
                g.move(y=innerRadius-leadInOut-rOffset)
                g.move(x=cutterRadius-depth)
                g.cut(y=radius+addendum+leadInOut-rOffset)
                g.move(x=cutterRadius+safeDistance)

else:
    g.Comment( '"%s" gearType is currently unsupported' )

# Program is done, shutdown time
g.append( 'M05 M09' )
g.append( 'G30' )
g.append( 'M30' )
g.append( '%' )

print g.output()

