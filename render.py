#!/usr/bin/python3

from math import sin, cos, radians
import re
import sys
import argparse
import statistics
import matplotlib.pyplot as plt
from celluloid import Camera
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate

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
parser.add_argument('--G', '-G', nargs=1, default='gear.svg', metavar='filename', help='Output SVG file')
parser.add_argument('--animate', '-a', action='store_true', help='Generate animation')
parser.add_argument('--picture', '-p', action='store_true', help='Generate picture')
parser.add_argument('--svg', '-g', action="store_true", help='Generate svg file')
parser.add_argument('--stats', '-s', action='store_true', help='Generate statistics')
parser.add_argument('--teeth', '-t', nargs=1, default=[-1], type=int, help='Number of teeth to draw')
args = parser.parse_args()

teethToDraw = args.teeth[0]
animationFile = args.A
pictureFile = args.P
svgFile = args.G
final = args.picture
animate = args.animate
svg = args.svg
verbose = args.verbose
stats = args.stats

if not (final or animate or svg or stats):
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
cuttings = []
with open('teeth.nc') as f:
    for line in f:
        l = line.strip()
        mTooth      = paramsTooth.match(l)
        mGeneral    = paramsGeneral.match(l)
        mTool       = paramsTool.match(l)
        mRotary     = paramsRotary.match(l)
        mgCode      = re.findall(parseGCode, l)

        # Tooth comment
        if mTooth:
            tooth = int(mTooth.group(1))
            if verbose:
                print('Processing tooth #%g' % tooth)

        # Numeric arguments
        elif mGeneral:
            name = mGeneral.group(1)
            value = float(mGeneral.group(2))
            locals()[name] = value

        # Rotary argument
        elif mRotary:
            rightRotary = mRotary.group(1) == 'True'

        # Tool argument
        elif mTool:
            toolAngle = float(mTool.group(1))
            toolDepth = float(mTool.group(2))
            toolRadius = float(mTool.group(3))
            toolTipHeight = float(mTool.group(4))

        # GCode
        elif mgCode:
            if stepNumber == 0:
                # Create all of the shapes the first time real GCode is encountered.
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
                direction = 1 if rightRotary else -1
                halfTip = toolTipHeight / 2.
                y = halfTip + sin(radians(toolAngle / 2.)) * toolDepth
                shaft = toolRadius - toolDepth
                cutter = Polygon([
                        (shaft, 4. * y),
                        (shaft, y),
                        (toolRadius, halfTip), 
                        (toolRadius, -halfTip),
                        (shaft, -y),
                        (-shaft, -y),
                        (-toolRadius, -halfTip),
                        (-toolRadius, halfTip),
                        (-shaft, y),
                        (-shaft, 4. * y),
                        ])

            # Move and cut based on each axis
            for axis, amt in mgCode:
                stepNumber += 1
                amt = float(amt)
                if axis == 'A':
                    gearBlank = rotate(gearBlank, curAngle - amt, origin = (0, 0))
                    curAngle = amt
                elif axis == 'X':
                    if cutterY:
                        curCutter = translate(cutter, cutterY, cutterZ)
                        areaStart = gearBlank.area
                        gearBlank = gearBlank.difference(curCutter)
                        # Deal with an acute cutter trimming off a shard
                        if type(gearBlank) == MultiPolygon:
                            gearBlank = gearBlank[0]
                        
                        # Track material removal
                        amountCut = areaStart - gearBlank.area
                        if amountCut > 0.:
                            cuttings.append(amountCut)

                        # Write an animation frame
                        if animate and (teethToDraw == -1 or tooth < teethToDraw):
                            plt.plot(*pitchCircle.exterior.xy, color='g')
                            plt.plot(*clearanceCircle.exterior.xy, color='c')
                            plt.plot(*gearBlank.exterior.xy, color='b')
                            plt.plot(*curCutter.exterior.xy, color='r')
                            plt.plot((0., cos(radians(-curAngle)) * outsideRadius), (0., sin(radians(-curAngle)) * outsideRadius), color='b')
                            plt.plot((-direction*(outsideRadius-hTotal), -direction*(outsideRadius-hTotal)), (-zMax, zMax), color='y')
                            plt.grid()
                            plt.axis('equal')
                            camera.snap()
                elif axis == 'Y':
                    cutterY = amt
                elif axis == 'Z':
                    cutterZ = amt

# Create the animation
if animate:
    if verbose:
        print('Generating animation "%s"' % animationFile)
    animation = camera.animate()
    animation.save(animationFile, writer = 'pillow')

# Create a final picture of the gear
if final:
    if verbose:
        print('Generating picture "%s"' % pictureFile)
    fig = plt.figure()
    plt.plot(*pitchCircle.exterior.xy, color='g')
    plt.plot(*clearanceCircle.exterior.xy, color='c')
    plt.plot(*dedendumCircle.exterior.xy, color='m')
    plt.plot(*gearBlank.exterior.xy, color='b')
    plt.grid()
    plt.axis('equal')
    plt.savefig(pictureFile)

# Create an svg file of only the gear
if svg:
    if verbose:
        print('Generating svg file "%s"' % svgFile)
    with open(svgFile, 'w') as f:
        f.write(gearBlank._repr_svg_())

# Print statistics
if stats:
    inches = True
    if inches:
        conv1, units1 = 1/ 25.4, "inch"
        conv2, units2 = conv1 / 25.4, "inch^2"
        conv3, units3 = conv2 / 25.4, "inch^3" 
    else:
        conv1, units1 = .1, "cm"
        conv2, units2 = .01, "cm^2"
        conv3, units3 = .001, "cc"

    print("Number of passes: %g" % len(cuttings))
    print("Total material removed: %g %s" % (conv3 * sum(cuttings) * blankThickness, units3))
    print("Cross section (per pass):")
    print("    Maximum: %g %s" % (conv2 * max(cuttings), units2))
    print("    Minimum: %g %s" % (conv2 * min(cuttings), units2))
    print("    Average: %g %s" % (conv2 * statistics.mean(cuttings), units2))
    print("Material removal (per pass):")
    print("    Maximum: %g %s" % (conv3 * blankThickness * max(cuttings), units3))
    print("    Minimum: %g %s" % (conv3 * blankThickness * min(cuttings), units3))
    print("    Average: %g %s" % (conv3 * blankThickness * statistics.mean(cuttings), units3))
    print("Cutting rate (per pass):")

    toolFeed = 200
    toolRPM = 4000
    toolFlutes = 4
    area = max(cuttings)
    cutTime = blankThickness / toolFeed
    cutCount = toolRPM * cutTime * toolFlutes
    materialPerFlute =  area * blankThickness / cutCount
    materialRR = area * blankThickness / cutTime

    print("    Time per each pass: %g mins" % cutTime)
    print("    Cuts per pass: %g" % cutCount)
    print("    Material per flute: %g %s" % (conv3 * materialPerFlute, units3))
    print("    Material removal rate: %g %s/min" % (conv3 * materialRR, units3))
