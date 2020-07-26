#!/usr/bin/python3

"""
G Code generator for cutting metric involute gears.

It is currently designed to use double angle shaft cutters or double bevel
cutters. In the future, the code will be modified to use single angle cutters
and slitting saws.

All input parameters are specified in millimeters or degrees.

Copyright 2020 - Michael Dubno - New York
"""

# FIX: Add multiple passes
# FIX: Add support for DP

import sys
from math import sin, cos, radians, degrees, sqrt, pi
import configargparse


def rotate(a, x, y):
    """Return point x,y rotated by angle a (in radians)."""
    return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)


class Tool():
    """The Tool class holds the specifications of the cutting tool."""

    def __init__(self, angle=40., depth=3., radius=10., tipHeight=0.,
                 number=1, rpm=2000, feed=200, mist=False, flood=False,
                 ease=0, mill='both'):
        assert angle >= 0., 'Tool: Angle must be greater than or equal to 0'
        assert depth > 0., 'Tool: Depth must be greater than 0'
        assert radius > depth, 'Tool: Radius must be greater than depth'
        assert tipHeight >= 0, 'Tool: tipHeight must be greater than or equal to 0'
        assert mill in ['both', 'climb', 'conventional'], 'Tool: mill must be "both", "climb", or "conventional'

        self.angle = radians(angle)
        self.depth = depth
        self.radius = radius
        self.tipHeight = tipHeight
        self.number = number
        self.rpm = rpm
        self.feed = feed
        self.mist = mist
        self.flood = flood
        self.ease = ease
        self.mill = mill

    def __str__(self):
        return "(Angle: {}, Depth: {}, Radius: {}, TipHeight: {})".format(
            degrees(self.angle), self.depth, self.radius, self.tipHeight)


class Gear():
    """The Gear class is used to generate G Code of involute gears."""

    def __init__(self, tool, module=1., pressureAngle=20., reliefFactor=1.25,
                 steps=5, cutterClearance=2., rightRotary=False):
        assert module > 0, 'Gear: Module must be greater than 0.'
        assert pressureAngle > 0, 'Gear: Pressure angle must be greater than 0.'
        assert steps >= 0, 'Gear: Steps must be greater than or equal to 0.'

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
G21 (Millimeters)
G30

T{number} G43 H{number} M6
S{rpm} M3{mist}{flood}
G54
F{feed}""".format(number=self.tool.number,
                  feed=self.tool.feed,
                  rpm=self.tool.rpm,
                  mist=' M07' if self.tool.mist else '',
                  flood=' M08' if self.tool.flood else '')

    def footer(self):
        return \
"""\
M5 M9
G30
M30
%"""

    def generate(self, teeth, blankThickness, teethToMake=0):
        assert teeth > 0, 'Gear: Number of teeth must be greater than 0.'
        assert blankThickness > 0, 'Gear: Blank thickness must be greater than 0.'

        # Calculate the variables used to generate the gear teeth
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
        mill = self.tool.mill
        angleDirection = 1 if self.rightRotary else -1
        xStart, xEnd = -angleDirection * xOffset, angleDirection * xOffset

        # Determine the maximum amount of height (or depth) in the Z axis before part of the cutter
        # won't intersect with the gear blank.
        zMax = min(sqrt(outsideRadius**2 - (outsideRadius-hAddendum)**2), outsideRadius * sin(radians(90.) - self.pressureAngle))
        zMax += zOffset
        zIncr = 0 if self.steps == 0 else zMax / self.steps

        # A partial number of teeth can be created if "teethToMake" is set,
        # otherwise all of the gears teeth are cut.
        if teethToMake == 0 or teethToMake > teeth:
            teethToMake = teeth

        # Make sure the cutter is big enough
        if hTotal > self.tool.depth:
           raise ValueError("Cutter depth is too shallow for tooth height")

        # Check to make sure the cutter shaft doesn't hit the gear blank.
        shaftRadius = self.tool.radius - self.tool.depth
        yP, zP = pitchRadius, zMax + zOffset
        yTool, zTool = rotate(-angleOffset, yP, zP)
        y = self.tool.radius + yTool - hDedendum
        shaftClearance = y - outsideRadius - shaftRadius
        if shaftClearance < 0:
            raise ValueError("Cutter shaft hits gear blank by %g mm" % -shaftClearance)

        # Include all of the generating parameters in the G Code header
        f = ['zMax', 'module', 'teeth', 'blankThickness', 'tool', 'reliefFactor',
             'pressureAngle', 'steps', 'cutterClearance', 'rightRotary', 'hAddendum',
             'hDedendum', 'hTotal', 'circularPitch', 'pitchDiameter', 'baseDiameter',
             'outsideDiameter', 'outsideRadius', 'zOffset', 'angleOffset', 'xStart',
             'xEnd']
        gcode = []
        for v in f:
            if v in locals():
                gcode.append('(%15s: %-70s)' % (v, locals()[v]))
            else:
                gcode.append('(%15s: %-70s)' % (v, getattr(self, v)))

        # Move to safe initial position
        cut = Cut(mill, xStart, xEnd, -angleDirection * self.cutterClearance)
        gcode.append('')
        gcode.append('G0 Z%g' % outsideRadius)
        gcode.append('G0 Y%g' % (-angleDirection * (outsideRadius + self.tool.radius + self.cutterClearance)))
        gcode.append(cut.start())

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

                    # Handle the special case of "easing into the first cut"
                    if self.tool.ease and zSteps == -self.steps:
                        yStart = self.tool.radius + yTool
                        yEnd = self.tool.radius + yTool - hDedendum
                        yDiv = (yEnd - yStart) / self.tool.ease
                        for easeStep in range(self.tool.ease):
                            y = yStart + yDiv * easeStep
                            gcode.append(cut.cut(
                                (angleDirection * degrees(angle + angleOffset + toothAngleOffset)),
                                (-angleDirection * y),
                                zTool))

                    gcode.append(cut.cut(
                        (angleDirection * degrees(angle + angleOffset + toothAngleOffset)),
                        (-angleDirection * (self.tool.radius + yTool - hDedendum)),
                        zTool))

                # Center of the slot
                if zSteps == 0:
                    gcode.append(cut.cut(
                        (angleDirection * degrees(angle + toothAngleOffset)),
                        (-angleDirection * (self.tool.radius + pitchRadius - hDedendum)),
                        z))

                # Top of the slot
                if zSteps >= 0:
                    yP, zP = pitchRadius, z - zOffset
                    yTool, zTool = rotate(-angleOffset, yP, zP)
                    gcode.append(cut.cut(
                        (angleDirection * degrees(angle - angleOffset + toothAngleOffset)),
                        (-angleDirection * (self.tool.radius + yTool - hDedendum)),
                        zTool))

        return '\n'.join(gcode)


class Cut():
    """Cut is used to generate the gcode for the actual cutting and retraction
    of the tool and the gear blank.  This can get a bit complicated due to the
    different styles of cutting (climb, conventional, both) and the various
    setups of the rotary table on the left or right side.
    """

    def __init__(self, mill, xStart, xEnd, yBackoff):
        self.mill = mill
        self.xStart = xStart
        self.xEnd = xEnd
        self.yBackoff = yBackoff
        self.stroke = 0

    def start(self):
        if self.mill == 'climb':
            return "G0 X%g" % self.xEnd
        return "G0 X%g" % self.xStart

    def cut(self, a, y, z):
        if self.mill == 'climb':
            ret = ["G1 X%g" % self.xStart,
                   "G0 Y%g" % (y + self.yBackoff),
                   "G0 X%g" % self.xEnd,
                   "G0 Y%g" % y]
        elif self.mill == 'conventional':
            ret = ["G1 X%g" % self.xEnd,
                   "G0 Y%g" % (y + self.yBackoff),
                   "G0 X%g" % self.xStart,
                   "G0 Y%g" % y]
        else:
            ret = ["G1 X%g" % [self.xStart, self.xEnd][self.stroke]]
            self.stroke = (self.stroke + 1) % 2

        return '\n'.join(["G0 A%g Y%g Z%g" % (a, y, z)] + ret)


def main():
    p = configargparse.ArgParser(
        default_config_files=['gears.cfg'],
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        prog="gears",
        description="Generate G Code to create involute spur gears.",
        epilog="""
            By using simple cutters like double angle shaft cutters or #8 gear cutter, you can cut
            most modules, pressure angles, and tooth count involute spur gears.  The limiting factors
            are the size of the cutter and the working envelope of the mill.

            A rotary 4th-axis is used to rotate the gear blank with the cutting tool held in the spindle.
                """)
    p.add('outfile', nargs='?', type=configargparse.FileType('w'), default=sys.stdout)
    p.add('--config', '-X', is_config_file=True, help='Config file path')

    # Tool arguments
    p.add('--angle', '-A', type=float, default=40., help='Tool: included angle in degrees')
    p.add('--depth', '-D', type=float, default=5., help='Tool: depth of cutting head in mm')
    p.add('--height', '-H', type=float, default=0., help='Tool: distance between the top and bottom of cutter at tip in mm')
    p.add('--diameter', '-I', type=float, default=15., help='Tool: cutting diameter at tip in mm')
    p.add('--number', '-N', type=int, default=1, help='Tool: tool number')
    p.add('--rpm', '-R', type=float, default=2000., help='Tool: spindle speed')
    p.add('--feed', '-F', type=float, default=200., help='Tool: feed rate')
    p.add('--mist', '-M', action='store_true', help='Tool: turn on mist coolant')
    p.add('--flood', '-L', action='store_true', help='Tool: turn on flood coolant')
    p.add('--ease', '-E', type=int, default=0, help='Tool: number of steps to "ease into" the first cut')
    p.add('--mill', default='both', choices=['both', 'climb', 'conventional'], help='Tool: cutting method')

    # Gear type arguments
    p.add('--module', '-m', type=float, default=1., help='Module of the gear')
    p.add('--pressure', '-p', type=float, default=20., help='Pressure angle in degrees')
    p.add('--relief', type=float, default=1.25, help='Relief factor (for the dedendum)')
    p.add('--steps', '-s', type=int, default=5, help='Steps/tooth face')
    p.add('--clear', '-c', type=float, default=2., help='Cutter clearance from gear blank in mm')
    p.add('--right', '-r', action='store_true', help='Rotary axis is on the right side of the machine')

    # Specific gear arguments
    p.add('--teeth', '-t', type=int, required=True, help='Number of teeth for the entire gear')
    p.add('--thick', '-k', type=float, required=True, help='Thickness of gear blank in mm')
    p.add('--make', type=int, default=0, help='Actual number of teeth to cut.')

    args = p.parse_args()

    try:
        tool = Tool(angle=args.angle, depth=args.depth, tipHeight=args.height,
                    radius=args.diameter / 2., number=args.number, rpm=args.rpm,
                    feed=args.feed, mist=args.mist, flood=args.flood, ease=args.ease,
                    mill=args.mill)

        g = Gear(tool, module=args.module, pressureAngle=args.pressure,
                 reliefFactor=args.relief, steps=args.steps, cutterClearance=args.clear,
                 rightRotary=args.right)

        print(g.header())
        print(g.generate(args.teeth, args.thick, args.make))
        print(g.footer())
    except Exception as e:
        print(e, file=sys.stderr)


if __name__ == '__main__':
    main()
