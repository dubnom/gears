from math import *

# Gear description
teeth           = 48
metric          = False     # True for metric (mm), False for imperial (in)
module          = .5
thickness       = .125      # Thickness of the gear surface (or edge of a crown)
spurGear        = False     # True for spur gears, False for crown gears

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
gearType        = 'Spur' if spurGear else 'Crown'
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

# Gcode helper to record a high-speed move operation
def gMove(a=None,x=None,y=None,z=None):
    gcode.append( "G0" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))

# Gcode helper to record a cutting speed linear operation
def gCut(a=None,x=None,y=None,z=None):
    gcode.append( "G1" + (' A%g'%a if a!=None else '') + (' X%g'%x if x!=None else '') + (' Y%g'%y if y!=None else '') + (' Z%g'%z if z!=None else ''))

# Gcode helper to record comments
def gComment(c=None):
    gcode.append( "(%s)" % c if c else '' )

# Preamble of parameter comments to assist machine setup
gcode = []
gcode.append( '%' )
gComment( "%s Triangle gear cutting" % gearType )
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
if spurGear:
    gMove(y=radius+cutterRadius+safeDistance)
    gMove(0.,leadInOut,z=0.)
    for tooth in range(teeth):
        gComment( 'Tooth %d' % tooth )
        for aOffset,rOffset,zOffset in anglePasses:
            gMove(tooth*anglePerTooth+aOffset,z=zOffset-centerlineOffset)
            for depth in depthPasses:
                gMove(y=radius+cutterRadius+addendum-rOffset-depth)
                gCut(x=-thickness-leadInOut)
                gMove(y=radius+cutterRadius+addendum+safeDistance)
                gMove(x=leadInOut)

else:   # Crown gear
    gMove(x=cutterRadius+safeDistance)
    gMove(0.,y=radius,z=0.)
    for tooth in range(teeth):
        gComment( 'Tooth %d' % tooth )
        for aOffset,rOffset,zOffset in anglePasses:
            gMove(tooth*anglePerTooth+aOffset,z=zOffset-centerlineOffset)
            for depth in depthPasses:
                gMove(y=innerRadius-leadInOut-rOffset)
                gMove(x=cutterRadius-depth)
                gCut(y=radius+addendum+leadInOut-rOffset)
                gMove(x=cutterRadius+safeDistance)

# Program is done, shutdown time
gcode.append( 'M05 M09' )
gcode.append( 'G30' )
gcode.append( 'M30' )

print '\n'.join(gcode)

