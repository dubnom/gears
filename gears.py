#!/usr/bin/python3

import argparse
import sys
from math import sin, cos, radians, degrees, sqrt, pi 

"""
G Code generator for cutting metric involute gears.

It is currently designed to use double angle shaft cutters or double bevel cutters.
In the future, the code will be modified to use single angle cutters and slitting saws.

All input parameters are specified in millimeters or degrees.
"""


def rotate(a, x, y):
    """Return point x,y rotated by angle a (in radians)."""
    return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)


# FIX: number of passes should be added
# FIX: coolant should be added
# FIX: Support for tool files and gear profile files should be added
# FIX: Add error checking if tool is too small
class Tool():
    """The Tool class holds the specifications of the cutting tool."""

    def __init__(self, angle=40., depth=3., radius=10., tipHeight=0., number=1, rpm=2000, feed=200):
        self.angle = radians(angle)
        self.depth = depth
        self.radius = radius
        self.tipHeight = tipHeight
        self.number = number
        self.rpm = rpm
        self.feed = feed

    def __str__(self):
        return "(Angle: %s, Depth: %s, Radius: %s, TipHeight: %s)" % (degrees(self.angle), self.depth, self.radius, self.tipHeight)


class Gear():
    """The Gear class is used to generate G Code of involute gears."""

    def __init__(self, tool, module=1., pressureAngle=20., reliefFactor=1.25, steps=5, cutterClearance = 2., rightRotary=False):
        self.module = module
        self.tool = tool
        self.reliefFactor = reliefFactor
        self.pressureAngle = radians(pressureAngle)
        self.steps = steps
        self.cutterClearance = cutterClearance
        self.rightRotary = rightRotary

    def header(self):
        return \
"""\
%
G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49
G21 (Millimeters)'
G30

T{number} G43 H{number} M6
S{rpm} M3 M9
G54
F{feed}""".format(number=self.tool.number, feed=self.tool.feed, rpm=self.tool.rpm)

    def footer(self):
        return \
"""\
M5 M9
G30
M30
%"""

    def generate(self, teeth, blankThickness, teethToMake=0):
        hAddendum = self.module
        hDedendum = self.module * self.reliefFactor
        hTotal = hAddendum + hDedendum
        circularPitch = self.module * pi
        pitchDiameter = self.module * teeth
        pitchRadius = pitchDiameter / 2.
        baseDiameter = pitchDiameter * cos(self.pressureAngle)
        outsideDiameter = pitchDiameter + 2 * hAddendum
        outsideRadius = outsideDiameter / 2.
        
        angleOffset = self.tool.angle / 2. - self.pressureAngle 
        zOffset = (circularPitch / 2. - 2. * sin(self.tool.angle / 2.) * hDedendum - self.tool.tipHeight) / 2.

        xOffset = self.cutterClearance + blankThickness / 2. + sqrt(self.tool.radius ** 2 - (self.tool.radius - hTotal) ** 2)
        xStart, xEnd = -xOffset, xOffset
        angleDirection = -1 if self.rightRotary else 1

        # Determine the maximum amount of height (or depth) in the Z axis before part of the cutter
        # won't intersect with the gear blank.
        zMax = min(sqrt(outsideRadius**2 - (outsideRadius-hAddendum)**2), outsideRadius * sin(radians(90.) - self.pressureAngle))
        zMax += zOffset
        zIncr = zMax / self.steps

        # FIX: Missing cleaning of the root
        # FIX: Steps may need to be spaced differently

        # A partial number of teeth can be created if "teethToMake" is set,
        # otherwise all of the gears teeth are cut.
        if teethToMake == 0 or teethToMake > teeth:
            teethToMake = teeth

        gcode = [] 

        # Include all of the generating parameters in the G Code header
        f = ['zMax', 'module', 'teeth', 'blankThickness', 'tool', 'reliefFactor', 'pressureAngle', 'steps',
                'cutterClearance', 'rightRotary', 'hAddendum', 'hDedendum', 'hTotal', 'circularPitch',
                'pitchDiameter', 'baseDiameter', 'outsideDiameter', 'outsideRadius', 'zOffset',
                'angleOffset', 'xStart', 'xEnd']
        for v in f:
            if v in locals():
                gcode.append('(%15s: %-70s)' % (v, locals()[v]))
            else:
                gcode.append('(%15s: %-70s)' % (v, getattr(self,v)))

        # Move to initial positions
        gcode.append('')
        x = xStart
        gcode.append('G0 X%g' % x)
        # FIX: Bring cutter to known safe position in Y and Z axes

        # Generate a tooth profile for ever tooth requested
        for tooth in range(teethToMake):
            toothAngleOffset = 2. * pi * tooth / teeth
            gcode.append('')
            gcode.append("( Tooth: %d)" % tooth)

            # The shape of the tooth (actually the space removed to make the tooth)
            # is created iteratively with a number of steps. More steps means greater
            # accuracy but longer run time.
            for zSteps in range(-self.steps, self.steps+1):
                z = zSteps * zIncr
                angle = z / pitchRadius

                # Bottom of the slot
                if zSteps <= 0:
                    yP, zP = pitchRadius, z + zOffset
                    yTool, zTool = rotate(angleOffset, yP, zP)
                    gcode.append('G0 Y%g A%g Z%g' % (
                            (-angleDirection * (self.tool.radius + yTool - hDedendum)),
                            (angleDirection * degrees(angle + angleOffset + toothAngleOffset)),
                            zTool))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)

                # Center of the slot
                if zSteps == 0:
                    gcode.append('G0 Y%g A%g Z%g' % ( 
                            (-angleDirection * (self.tool.radius + pitchRadius - hDedendum)),
                            (angleDirection * degrees(angle + toothAngleOffset)),
                            z))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)
                
                # Top of the slot
                if zSteps >= 0:
                    yP, zP = pitchRadius, z - zOffset
                    yTool, zTool = rotate(-angleOffset, yP, zP)
                    gcode.append('G0 Y%g A%g Z%g' % (
                            (-angleDirection * (self.tool.radius + yTool - hDedendum)),
                            (angleDirection * degrees(angle - angleOffset + toothAngleOffset)),
                            zTool))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)

        return '\n'.join(gcode)


def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            prog="gears",
            description="Generate G Code to create involute spur gears.",
            epilog="""
                By using simple cutters like double angle shaft cutters or #8 gear cutter, you can cut
                most modules, pressure angles, and tooth count involute spur gears.  The limiting factors
                are the size of the cutter and the working envelope of the mill.

                A rotary 4th-axis is used to rotate the gear blank with the cutting tool held in the spindle.
                """)
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('w'), default=sys.stdout)

    # Tool arguments
    parser.add_argument('--angle', '-A', type=float, default=40., help='Tool: included angle in degrees')
    parser.add_argument('--depth', '-D', type=float, default=5., help='Tool: depth of cutting head in mm')
    parser.add_argument('--height', '-H', type=float, default=0., help='Tool: distance between the top and bottom of cutter at tip in mm')
    parser.add_argument('--diameter', '-I', type=float, default=15., help='Tool: cutting diameter at tip in mm')
    parser.add_argument('--number', '-N', type=int, default=1, help='Tool: tool number')
    parser.add_argument('--rpm', '-R', type=float, default=2000., help='Tool: spindle speed')
    parser.add_argument('--feed', '-F', type=float, default=200., help='Tool: feed rate')
    # Coolant

    # Gear type arguments
    parser.add_argument('--module', '-m', type=float, default=1., help='Module of the gear')
    parser.add_argument('--pressure', '-p', type=float, default=20., help='Pressure angle in degrees')
    parser.add_argument('--relief', type=float, default=1.25, help='Relief factor (for the dedendum)')
    parser.add_argument('--steps', '-s', type=int, default=5, help='Steps/tooth face')
    parser.add_argument('--clear', '-c', type=float, default=2., help='Cutter clearance from gear blank in mm')
    # Rotary

    # Specific gear arguments
    parser.add_argument('--teeth', '-t', type=int, required=True, help='Number of teeth for the entire gear')
    parser.add_argument('--thick', '-k', type=float, required=True, help='Thickness of gear blank in mm')
    parser.add_argument('--make', type=int, default=0, help='Actual number of teeth to cut.')

    args = parser.parse_args()
    print(args)

 
    tool = Tool(angle=args.angle, depth=args.depth, tipHeight=args.height, radius=args.diameter / 2.,
            number=args.number, rpm=args.rpm, feed=args.feed)
    g = Gear(tool, module=args.module, pressureAngle=args.pressure,
            reliefFactor=args.relief, steps=args.steps, cutterClearance=args.clear,
            rightRotary=True)

    print(g.header())
    print(g.generate(args.teeth, args.thick, args.make))
    print(g.footer())


if __name__ == '__main__':
    main()
