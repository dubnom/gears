#!/usr/bin/python3

"""
Render 2-dimensional pictures and animations from gear cutting G code files.

Also calculate and display statistics about the cutting process.

Copyright 2020 - Michael Dubno - New York
"""

from math import sin, cos, radians, degrees, pi
import statistics
import re
import sys
import configargparse
import matplotlib.pyplot as plt
from celluloid import Camera
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate, translate

# Parse the command line arguments
p = configargparse.ArgParser(
    default_config_files=['render.cfg'],
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    prog="render",
    description="Render G Code gear cutting files.",
    epilog="""
        Render can create animations and/or final images from a G Code involute
        spur gear cutting file.  Creating animated GIFS can take a long time,
        so you can speed things up by animating a limited set of teeth.
        """)
p.add('infile', nargs='?', type=configargparse.FileType('r'), default=sys.stdin)
p.add('--verbose', '-v', action='count', default=0, help='Show progress messages')
p.add('--A', '-A', nargs=1, default='animation.gif', metavar='filename', help='Output animation file')
p.add('--P', '-P', nargs=1, default='picture.png', metavar='filename', help='Output picture file')
p.add('--G', '-G', nargs=1, default='gear.svg', metavar='filename', help='Output SVG file')
p.add('--animate', '-a', action='store_true', help='Generate animation')
p.add('--picture', '-p', action='store_true', help='Generate picture')
p.add('--svg', '-g', action='store_true', help='Generate svg file')
p.add('--stats', '-s', action='store_true', help='Generate statistics')
p.add('--inches', '-i', action='store_true', help='Show statistics in imperial units')
p.add('--teeth', '-t', nargs=1, default=[-1], type=int, help='Number of teeth to draw')
args = p.parse_args()

teeth_to_draw = args.teeth[0]
animationFile = args.A
pictureFile = args.P
svg_file = args.G
final = args.picture
animate = args.animate
svg = args.svg
verbose = args.verbose
stats = args.stats
inches = args.inches

if not (final or animate or svg or stats):
    parser.print_help()
    exit(-1)


# Regular expressions used to parse file from gears.py
parse_tooth = re.compile(r'^\( Tooth: ([-0-9]+)\)$')
parse_general = re.compile(r'^\( *([a-z_A-Z]+): ([-0-9\.]+) *\)$')
parse_rotary = re.compile(r'^\( *right_rotary: (True|False) *\)$')
parse_tool = re.compile(r'^\( *tool: \(Angle: ([0-9\.]+), Depth: ([0-9\.]+), Radius: ([0-9\.]+), TipHeight: ([0-9\.]+), Flutes: ([0-9]+)\) *\)$')
parse_feed = re.compile(r'^F([-0-9\.]+)')
parse_speed = re.compile(r'^S([-0-9\.]+)')
parse_gcode = r'([AXYZ])([-0-9\.]+)'

# Start up a camera if we're creating an animation
if animate:
    fig = plt.figure()
    camera = Camera(fig)

# Run through each line of the file
cutter_y = cutter_z = cur_angle = 0.
tooth = 0
step_number = 0
cuttings = []
v = {}
with open('teeth.nc') as f:
    for line in f:
        l = line.strip()
        mTooth = parse_tooth.match(l)
        mGeneral = parse_general.match(l)
        mTool = parse_tool.match(l)
        mRotary = parse_rotary.match(l)
        mFeed = parse_feed.match(l)
        mSpeed = parse_speed.match(l)
        mgCode = re.findall(parse_gcode, l)


        # Tooth comment
        if mTooth:
            tooth = int(mTooth.group(1))
            if verbose:
                print('Processing tooth #%g' % tooth)

        # Numeric arguments
        elif mGeneral:
            name = mGeneral.group(1)
            value = float(mGeneral.group(2))
            v[name] = value

        # Rotary argument
        elif mRotary:
            right_rotary = mRotary.group(1) == 'True'

        # Tool argument
        elif mTool:
            tool_angle = float(mTool.group(1))
            tool_depth = float(mTool.group(2))
            tool_radius = float(mTool.group(3))
            tool_tip_height = float(mTool.group(4))
            tool_flutes = int(mTool.group(5))

        # Feed rate
        elif mFeed:
            tool_feed = float(mFeed.group(1))

        # Spindle speed
        elif mSpeed:
            tool_rpm = float(mSpeed.group(1))

        # GCode
        elif mgCode:
            if step_number == 0:
                # Create all of the shapes the first time real GCode is encountered.
                if verbose:
                    print('Header has been read')

                # Create a polygon to represent the gear blank
                r = v['outside_diameter'] / 2.
                gear_blank = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the pitch circle
                r = v['outside_diameter'] / 2. - v['h_addendum']
                pitch_circle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the dedendum circle
                r = v['outside_diameter'] / 2. - v['h_addendum'] - v['h_addendum']
                dedendum_circle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to for the clearance circle
                r = v['outside_diameter'] / 2. - v['h_total']
                clearance_circle = Polygon([(r*cos(radians(a)), r*sin(radians(a))) for a in range(0, 360, 1)])

                # Create a polygon to represent the cutting tool
                direction = 1 if right_rotary else -1
                half_tip = tool_tip_height / 2.
                y = half_tip + sin(radians(tool_angle / 2.)) * tool_depth
                shaft = tool_radius - tool_depth
                cutter = Polygon([
                    (shaft, v['outside_radius']),
                    (shaft, y),
                    (tool_radius, half_tip),
                    (tool_radius, -half_tip),
                    (shaft, -y),
                    (-shaft, -y),
                    (-tool_radius, -half_tip),
                    (-tool_radius, half_tip),
                    (-shaft, y),
                    (-shaft, v['outside_radius']),
                    ])

            # Move and cut based on each axis
            for axis, amt in mgCode:
                step_number += 1
                amt = float(amt)
                if axis == 'A':
                    gear_blank = rotate(gear_blank, cur_angle - amt, origin=(0, 0))
                    cur_angle = amt
                elif axis == 'X':
                    if cutter_y:
                        cur_cutter = translate(cutter, cutter_y, cutter_z)
                        area_start = gear_blank.area
                        gear_blank = gear_blank.difference(cur_cutter)
                        # Deal with an acute cutter trimming off a shard
                        if type(gear_blank) == MultiPolygon:
                            big_poly, area = None, 0.
                            for polygon in gear_blank:
                                if polygon.area > area:
                                    big_poly, area = polygon, polygon.area
                            gear_blank = big_poly

                        # Track material removal
                        amountCut = area_start - gear_blank.area
                        if amountCut > 0.:
                            cuttings.append(amountCut)

                        # Write an animation frame
                        if animate and (teeth_to_draw == -1 or tooth < teeth_to_draw):
                            plt.plot(*pitch_circle.exterior.xy, color='g')
                            plt.plot(*clearance_circle.exterior.xy, color='c')
                            plt.plot(*gear_blank.exterior.xy, color='b')
                            plt.plot(*cur_cutter.exterior.xy, color='r')
                            plt.plot((0., cos(radians(-cur_angle)) * v['outside_radius']), (0., sin(radians(-cur_angle)) * v['outside_radius']), color='b')
                            plt.plot((-direction*(v['outside_radius'] - v['h_total']), -direction*(v['outside_radius'] - v['h_total'])), (-v['z_max'], v['z_max']), color='y')
                            plt.grid()
                            plt.axis('equal')
                            camera.snap()
                elif axis == 'Y':
                    cutter_y = amt
                elif axis == 'Z':
                    cutter_z = amt

# Create the animation
if animate:
    if verbose:
        print('Generating animation "%s"' % animationFile)
    animation = camera.animate()
    animation.save(animationFile, writer='pillow')

# Create a final picture of the gear
if final:
    if verbose:
        print('Generating picture "%s"' % pictureFile)
    fig = plt.figure()
    plt.plot(*pitch_circle.exterior.xy, color='g')
    plt.plot(*clearance_circle.exterior.xy, color='c')
    plt.plot(*dedendum_circle.exterior.xy, color='m')
    plt.plot(*gear_blank.exterior.xy, color='b')
    plt.grid()
    plt.axis('equal')
    plt.savefig(pictureFile)

# Create an svg file of only the gear
if svg:
    if verbose:
        print('Generating svg file "%s"' % svg_file)
    with open(svg_file, 'w') as f:
        f.write(gear_blank._repr_svg_())

# Print statistics
if stats:
    area = max(cuttings)
    cut_time = v['blank_thickness'] / tool_feed
    cut_count = tool_rpm * cut_time * tool_flutes
    materialPerFlute = area * v['blank_thickness'] / cut_count
    materialRR = area * v['blank_thickness'] / cut_time

    surfaceMPS = .001 * tool_radius * 2. * pi * (tool_rpm / 60.)

    if inches:
        conv1, units1 = 1/ 25.4, "inch"
        conv2, units2 = conv1 / 25.4, "inch^2"
        conv3, units3 = conv2 / 25.4, "inch^3"
    else:
        conv1, units1 = 1, "mm"
        conv2, units2 = .01, "cm^2"
        conv3, units3 = .001, "cc"

    print("Parameters:")
    print("    Tool:")
    print("        RPM: %g" % tool_rpm)
    print("        Radius: %g %s" % (conv1 * tool_radius, units1))
    print("        Depth: %g %s" % (conv1 * tool_depth, units1))
    print("        Flutes: %g" % tool_flutes)
    print("        Feed: %g %s/minute" % (conv1 * tool_feed, units1))
    print("        Feed per tooth: %g %s" % (conv1 * tool_feed / (tool_rpm * tool_flutes), units1))
    if inches:
        print("        SFM: %g feet/minute" % (surfaceMPS / .00508))
    else:
        print("        Surface meters/minute: %g" % (surfaceMPS * 60.))
    print("    Gear:")
    print("        Module: %g" % v['module'])
    print("        Teeth: %g" % v['teeth'])
    print("        Thickness: %g %s" % (v['blank_thickness'] * conv1, units1))
    print("        Pressure Angle: %g degrees" % degrees(v['pressure_angle']))
    print("    Cutting:")
    print("        Passes: %g" % len(cuttings))
    print("        Total material removed: %g %s" % (conv3 * sum(cuttings) * v['blank_thickness'], units3))
    print("Cross section (per pass):")
    print("    Maximum: %g %s" % (conv2 * max(cuttings), units2))
    print("    Minimum: %g %s" % (conv2 * min(cuttings), units2))
    print("    Average: %g %s" % (conv2 * statistics.mean(cuttings), units2))
    print("Material removal (per pass):")
    print("    Maximum: %g %s" % (conv3 * v['blank_thickness'] * max(cuttings), units3))
    print("    Minimum: %g %s" % (conv3 * v['blank_thickness'] * min(cuttings), units3))
    print("    Average: %g %s" % (conv3 * v['blank_thickness'] * statistics.mean(cuttings), units3))
    print("Cutting rate (per pass):")
    print("    Time per each pass: %g mins" % cut_time)
    print("    Cuts per pass: %g" % cut_count)
    print("    Material per flute: %g %s" % (conv3 * materialPerFlute, units3))
    print("    Material removal rate: %g %s/min" % (conv3 * materialRR, units3))
