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


# FIX: tool.radius must be added/subtracted from the Y value!!!
class Tool():
    """The Tool class holds the specifications of the cutting tool."""

    def __init__(self, angle=40., depth=3., radius=10., tipHeight=0.):
        self.angle = radians(angle)
        self.depth = depth
        self.radius = radius
        self.tipHeight = tipHeight

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
                'cutterClearance', 'rightRotary', 'hAddendum', 'hDedendum', 'hTotal', 'circularPitch', 'pitchDiameter', 'baseDiameter',
                'outsideDiameter', 'outsideRadius', 'zOffset', 'angleOffset', 'xStart', 'xEnd']
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
                            (-angleDirection * (yTool - hDedendum)),
                            (angleDirection * degrees(angle + angleOffset + toothAngleOffset)),
                            zTool))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)

                # Center of the slot
                if zSteps == 0:
                    gcode.append('G0 Y%g A%g Z%g' % ( 
                            (-angleDirection * (pitchRadius - hDedendum)),
                            (angleDirection * degrees(angle + toothAngleOffset)),
                            z))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)
                
                # Top of the slot
                if zSteps >= 0:
                    yP, zP = pitchRadius, z - zOffset
                    yTool, zTool = rotate(-angleOffset, yP, zP)
                    gcode.append('G0 Y%g A%g Z%g' % (
                            (-angleDirection * (yTool - hDedendum)),
                            (angleDirection * degrees(angle - angleOffset + toothAngleOffset)),
                            zTool))
                    x = xEnd if x == xStart else xStart
                    gcode.append("G1 X%g" % x)

        return '\n'.join(gcode)


def main():
    g = Gear(Tool(angle=40., depth=4.), rightRotary=True, steps=15)

    print('%')
    print('G90 G54 G64 G50 G17 G40 G80 G94 G91.1 G49')
    print('G21 (Millimeters)')
    print('G30')

    print('T40 G43 H40 M6')
    print('S4000 M3 M9')
    print('G54')
    print('F200')

    print(g.generate(7, 22., teethToMake=0))

    print('M5 M9')
    print('G30')
    print('M30')
    print('%')


if __name__ == '__main__':
    main()
