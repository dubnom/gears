from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
import matplotlib.pyplot as plt
from math import *
import re
from celluloid import Camera


animate = True 
final   = True

# Regular expressions used to parse file from gears.py
paramsTooth     = re.compile('^\( Tooth: ([-0-9]+)\)$')
paramsGeneral   = re.compile('^\( *([a-zA-Z]+): ([-0-9\.]+) *\)$')
paramsRotary    = re.compile('^\( *rightRotary: (True|False) *\)$')
paramsTool      = re.compile('^\( *tool: \(Angle: ([-0-9\.]+), Depth: ([-0-9\.]+), Radius: ([-0-9\.]+), TipHeight: ([-0-9\.]+)\) *\)$') 
gCode = re.compile('^G([01]+) ([AXYZ])([-0-9\.]+)$')

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
        mgCode      = gCode.match(l)

        if mTooth:
            tooth = int(mTooth.group(1))

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

            stepNumber += 1
            axis = mgCode.group(2)
            amt = float(mgCode.group(3))
            if axis == 'A':
                gearBlank = rotate(gearBlank, curAngle - amt)
                curAngle = amt
            elif axis == 'X':
                if cutterY:
                    curCutter = translate(cutter, cutterY + outsideRadius, cutterZ)
                    gearBlank = gearBlank.difference(curCutter)

                    # Write an animation frame
                    if animate and tooth < 3:
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
    animation = camera.animate()
    animation.save('animation.gif', writer = 'imagemagick')

# Create a final picture of the gear (if we're creating final pictures of gears)
if final:
    plt.plot(*gearBlank.exterior.xy)
    plt.plot(*pitchCircle.exterior.xy, color='g')
    plt.plot(*dedendumCircle.exterior.xy, color='m')
    plt.plot(*clearanceCircle.exterior.xy, color='c')
    #plt.plot(*curCutter.exterior.xy)
    #plt.plot((0., cos(radians(curAngle)) * outsideDiameter / 2.), (0., sin(radians(curAngle)) * outsideDiameter / 2.))
    plt.grid()
    plt.axis('equal')
    plt.savefig('final.png')

