from math import *

teeth           = 16
metric          = False
module          = .5
thickness       = .125

toolNumber      = 10
spindleSpeed    = 3000
feedRate        = 24.
cutterDiameter  = .75
cutterAngle     = 60.
centerlineOffset= .100      # distance from bottom of tool to edge cutting point

depthOfCut      = .04
yClearance      = .1
xClearance      = .2

module          = module if metric else module / 25.4
units           = 'mms' if metric else 'inches'
diameter        = teeth * module
radius          = diameter / 2.
circumference   = diameter * pi
toothHeight     = sin(radians(cutterAngle)) * circumference/teeth
addendum        = toothHeight / 2.
dedendum        = toothHeight / 2.
extraAngle      = 180./teeth
cutterRadius    = cutterDiameter / 2.


# Angle passes are offsets to a centerline cutting setup to allow a standard fixed size cutter to create
# included angles larger than the cutter's natural angle.  This is done by slightly rotating the gear up and down
# while adjusting the height and y distance from the blank.
innerRadius = radius - dedendum
anglePasses = [
        # aOffset,    yOffset,                                            zOffset
        (-extraAngle, innerRadius - cos(radians(extraAngle))*innerRadius, -sin(radians(extraAngle))*innerRadius),
        (extraAngle,  innerRadius - cos(radians(extraAngle))*innerRadius, sin(radians(extraAngle))*innerRadius)
]

# Passes are the depths that the cutter should use for cutting the teeth
passes = []
depth = 0.
while depth < toothHeight:
    depth = min(depth+depthOfCut, toothHeight)
    passes.append(depth)

# Gcode helper to record a high-speed move operation
def gMove(a=None,x=None,y=None,z=None):
    gcode.append( "G0" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))

# Gcode helper to record a cutting speed linear operation
def gCut(a=None,x=None,y=None,z=None):
    gcode.append( "G1" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))

# Gcode helper to record comments
def gComment(c=None):
    gcode.append( "(%s)" % c if c else '' )

gcode = []
gcode.append( '%' )
gComment( "Triangle gear cutting" )
gComment( "Teeth: %d,  Module: %g %s,  Thickness: %g %s" % (teeth, module, units, thickness, units))
gComment()
gComment( "Diameter: %g %s" % (diameter, units))
gComment( "Radius: %g %s" % (radius, units))
gComment( "Circumference: %g %s" % (circumference, units))
gComment( "ToothHeight: %g %s" % (toothHeight, units))
gComment( "Gear Blank Diameter: %g %s" % (diameter + 2.*addendum, units))
gComment( "ExtraAngle: %g degrees" % extraAngle )
gComment()

# Setup the machine, choose the tool, set the rates
gComment( 'T%d D=%g - Dual bevel %g degree cutter' % (toolNumber, cutterDiameter, cutterAngle))
gcode.append( 'G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49' )
gcode.append( 'G21 (mm)' if metric else 'G20 (inch)' )
gcode.append( 'G30' )
gcode.append( 'T%d G43 H%d M6' % (toolNumber, toolNumber))
gcode.append( 'S%d M3 M8' % spindleSpeed )
gcode.append( 'G54' )
gcode.append( 'F%g' % feedRate )

# Actual logic to loop through each tooth, angle pass, and depth pass generating Gcode
anglePerTooth = 360./teeth
gMove(y=radius+cutterRadius+yClearance)
gMove(0.,xClearance,z=0.)
for tooth in range(teeth):
    gComment( 'Tooth %d' % tooth )
    for aOffset,yOffset,zOffset in anglePasses:
        gMove(tooth*anglePerTooth+aOffset,z=zOffset-centerlineOffset)
        for p in passes:
            gMove(y=radius+cutterRadius+addendum-yOffset-p)
            gCut(x=-thickness-xClearance)
            gMove(y=radius+cutterRadius+addendum+yClearance)
            gMove(x=xClearance)

# Program is done, shutdown time
gcode.append( 'M05 M09' )
gcode.append( 'G30' )
gcode.append( 'M30' )

print '\n'.join(gcode)

