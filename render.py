#!/usr/bin/python3

from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt
from math import sin, cos, radians
import re
import sys
from celluloid import Camera
import argparse

# Parse the command line arguments
parser = argparse.ArgumentParser(
        prog="render",
        description="Render G Code gear cutting files.",
        epilog="""
            Render can create animations and/or final images from a G Code involute
            spur gear cutting file.  Creating animated GIFS can take a long time,
            so you can speed things up by animating a limited set of teeth.
            """)
parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
parser.add_argument('--verbose', '-v', action='count', default=0, help='Show progress messages')
parser.add_argument('--A', '-A', nargs=1, default='animation.gif', metavar='filename', help='Output animation file')
parser.add_argument('--P', '-P', nargs=1, default='picture.png', metavar='filename', help='Output picture file')
parser.add_argument('--animate', '-a', action='count', default=0, help='Generate animation')
parser.add_argument('--picture', '-p', action='count', default=0, help='Generate picture')
parser.add_argument('--teeth', nargs=1, default=[-1], type=int, help='Number of teeth to draw')
args = parser.parse_args()

teethToDraw = args.teeth[0]
animationFile = args.A
pictureFile = args.P
final = args.picture > 0
animate = args.animate > 0
verbose = args.verbose


if not (final or animate):
    parser.print_help()
    exit(-1)


# Regular expressions used to parse file from gears.py
paramsTooth     = re.compile('^\( Tooth: ([-0-9]+)\)$')
paramsGeneral   = re.compile('^\( *([a-zA-Z]+): ([-0-9\.]+) *\)$')
paramsRotary    = re.compile('^\( *rightRotary: (True|False) *\)$')
paramsTool      = re.compile('^\( *tool: \(Angle: ([-0-9\.]+), Depth: ([-0-9\.]+), Radius: ([-0-9\.]+), TipHeight: ([-0-9\.]+)\) *\)$') 
parseGCode      = '([AXYZ])([-0-9\.]+)'

# Start up a camera if we're creating an animation
if animate:
    fig = plt.figure()
    camera = Camera(fig)

# Run through each line of the file
cutterY = cutterZ = curAngle = 0.
tooth = 0
stepNumber = 0
with open('teeth.nc') as f:
    for line in f:
        l = line.strip()
        mTooth      = paramsTooth.match(l)
        mGeneral    = paramsGeneral.match(l)
        mTool       = paramsTool.match(l)
        mRotary     = paramsRotary.match(l)
        mgCode      = re.findall(parseGCode, l)

        if mTooth:
            tooth = int(mTooth.group(1))
            if verbose:
                print('Processing tooth #%g' % tooth)

        elif mGeneral:
            name = mGeneral.group(1)
            value = float(mGeneral.group(2))
            locals()[name] = value

        elif mRotary:
            rightRotary = mRotary.group(1) == 'True'

        elif mTool:
            toolAngle = float(mTool.group(1))
            toolDepth = float(mTool.group(2))
            toolRadius = float(mTool.group(3))
            toolTipHeight = float(mTool.group(4))

        elif mgCode:
            if stepNumber == 0:
                if verbose:
                    print('Header has been read')

                # Create a polygon to represent the gear blank
                r = outsideDiameter / 2.
                gearBlank = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the pitch circle
                r = outsideDiameter / 2. - hAddendum
                pitchCircle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the dedendum circle
                r = outsideDiameter / 2. - hAddendum - hAddendum
                dedendumCircle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the clearance circle
                r = outsideDiameter / 2. - hTotal
                clearanceCircle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to represent the cutting tool
                y = toolTipHeight / 2. + sin(radians(toolAngle / 2.)) * toolDepth
                cutter = Polygon([
                        (0, toolTipHeight / 2.),
                        (toolDepth, y), 
                        (toolDepth, -y),
                        (0, -toolTipHeight / 2.),
                        ])

            for axis, amt in mgCode:
                stepNumber += 1
                amt = float(amt)
                if axis == 'A':
                    gearBlank = rotate(gearBlank, curAngle - amt, origin = (0, 0))
                    curAngle = amt
                elif axis == 'X':
                    if cutterY:
                        curCutter = translate(cutter, cutterY, cutterZ)
                        gearBlank = gearBlank.difference(curCutter)
                        # Deal with an acute cutter trimming off a shard
                        if type(gearBlank) == MultiPolygon:
                            gearBlank = gearBlank[0]

                        # Write an animation frame
                        if animate and (teethToDraw == -1 or tooth < teethToDraw):
                            plt.plot(*pitchCircle.exterior.xy, color='g')
                            plt.plot(*clearanceCircle.exterior.xy, color='c')
                            plt.plot(*gearBlank.exterior.xy, color='b')
                            plt.plot(*curCutter.exterior.xy, color='r')
                            plt.plot((0., cos(radians(-curAngle)) * outsideRadius), (0., sin(radians(-curAngle)) * outsideRadius), color='b')
                            plt.plot((outsideRadius-hTotal, outsideRadius-hTotal), (-zMax, zMax), color='y')
                            plt.grid()
                            plt.axis('equal')
                            camera.snap()
                elif axis == 'Y':
                    cutterY = amt
                elif axis == 'Z':
                    cutterZ = amt

# Create the animation (if we're creating animations)
if animate:
    if verbose:
        print('Generating animation "%s"' % animationFile)
    animation = camera.animate()
    animation.save(animationFile, writer = 'pillow')

# Create a final picture of the gear (if we're creating final pictures of gears)
if final:
    if verbose:
        print('Generating picture "%s"' % pictureFile)
    plt.plot(*pitchCircle.exterior.xy, color='g')
    plt.plot(*clearanceCircle.exterior.xy, color='c')
    plt.plot(*dedendumCircle.exterior.xy, color='m')
    plt.plot(*gearBlank.exterior.xy, color='b')
    plt.grid()
    plt.axis('equal')
    plt.savefig(pictureFile)

