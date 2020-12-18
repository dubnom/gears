#!/usr/bin/python3

"""
Render 2-dimensional pictures and animations from gear cutting G code files.

Also calculate and display statistics about the cutting process.

Copyright 2020 - Michael Dubno - New York
"""

from math import sin, cos, radians, degrees, pi, tan, tau
import statistics
import re
import sys
import configargparse
import matplotlib.pyplot as plt
from celluloid import Camera
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import rotate, translate

# Parse the command line arguments
import gear_plot
from anim.geom import BBox
from anim.simple import SimpleAnimation
from gear_cycloidal import CycloidalPair
from rack import Rack
from tool import Tool


def circle(radius):
    """Generate a circle Polygon() with radius"""
    return Polygon([(radius * cos(radians(a)), radius * sin(radians(a))) for a in range(0, 360, 1)])


p = configargparse.ArgParser(
    default_config_files=['render.cfg'],
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    description="Render G Code gear cutting files.",
    epilog="""
        Render can create animations and/or pictures from a G Code involute
        spur gear cutting file.  Creating animated GIFS can take a long time,
        so you can speed things up by animating a limited set of teeth.
        """)
p.add('infile', nargs='?', type=configargparse.FileType('r'), default=sys.stdin)
p.add('--verbose', '-v', action='count', default=0, help='Show progress messages')
p.add('--A', '-A', nargs=1, default='animation.gif', metavar='filename', help='Output animation file')
p.add('--P', '-P', nargs=1, default='picture.png', metavar='filename', help='Output picture file')
p.add('--G', '-G', nargs=1, default='gear.svg', metavar='filename', help='Output SVG file')
p.add('--zerror', '-Z', type=float, default=0., help='Z-axis error in mm')
p.add('--yerror', '-Y', type=float, default=0., help='Y-axis error in mm')
p.add('--animate', '-a', action='store_true', help='Generate animation')
p.add('--picture', '-p', action='store_true', help='Generate picture')
p.add('--svg', '-g', action='store_true', help='Generate svg file')
p.add('--stats', '-s', action='store_true', help='Generate statistics')
p.add('--zoom', '-z', action='count', help='Zoom into two teeth in picture (-zz for tighter zoom)')
p.add('--inches', '-i', action='store_true', help='Show statistics in imperial units')
p.add('--teeth', '-t', nargs=1, default=[-1], type=int, help='Number of teeth to draw')
p.add('--first', '-f', default=0, type=int, help='First tooth to draw')
args = p.parse_args()

if args.teeth[0] != -1:
    teeth_to_draw = set(range(args.first, args.first + args.teeth[0]))
else:
    teeth_to_draw = 'all'
animationFile = args.A
pictureFile = args.P
svg_file = args.G
picture = args.picture
animate = args.animate
svg = args.svg
verbose = args.verbose
stats = args.stats
inches = args.inches
infile = args.infile
zoom = args.zoom
zerror = args.zerror
yerror = args.yerror

if not (picture or animate or svg or stats):
    p.print_help()
    exit(-1)

def log_cut(*args):
    do_log = False
    if do_log:
        print(*args)


# Regular expressions used to parse file from gears.py
parse_tooth = re.compile(r'^\( Tooth: ([-0-9]+) *\)$')
parse_general = re.compile(r'^\( *([a-z_A-Z]+): ([-0-9\.]+) *\)$')
parse_rotary = re.compile(r'^\( *right_rotary: (True|False) *\)$')
parse_tool = re.compile(r'^\( *tool: \(%s\) *\)$' % ', '.join(r'%s: ([0-9\.]+)' % f for f in ['Angle', 'Depth', 'Radius', 'TipHeight', 'Extension', 'Flutes']))
parse_feed = re.compile(r'^F([-0-9\.]+)')
parse_speed = re.compile(r'^S([-0-9\.]+)')
parse_movecut = re.compile(r'^G([01]+)')
parse_gcode = r'([AXYZ])([-0-9\.]+)'
parse_tool_details = re.compile(r'^\( *ToolDetails: ({.*}) *\)$')

# Start up a camera if we're creating an animation
if animate:
    fig = plt.figure()
    camera = Camera(fig)
    sa = SimpleAnimation(None, image_size=(500, 500))
else:
    sa = None

# Run through each line of the file
cutter_y = cutter_z = cur_angle = 0.
tooth = 0
step_number = 0
cuttings = []
v = {}
tooth_last = -1

for line_number, line in enumerate(infile):
    l = line.strip()
    mTooth = parse_tooth.match(l)
    mGeneral = parse_general.match(l)
    # mTool = parse_tool.match(l)
    mTool = None
    mToolDetails = parse_tool_details.match(l)
    mRotary = parse_rotary.match(l)
    mFeed = parse_feed.match(l)
    mSpeed = parse_speed.match(l)
    mMoveCut = parse_movecut.match(l)
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
        # tool_radius += 0.1
        tool_tip_height = float(mTool.group(4))
        tool_shaft_extension = float(mTool.group(5))
        tool_flutes = int(mTool.group(6))

    # ToolDetails argument
    elif mToolDetails:
        tool = Tool.from_json(mToolDetails.group(1))
        print('NewTool: %r' % tool)
        tool_angle = tool.angle_degrees
        tool_depth = tool.depth
        tool_radius = tool.radius
        # tool_radius += 0.1
        tool_tip_height = tool.tip_height
        tool_shaft_extension = tool.shaft_extension
        tool_flutes = tool.flutes

    # Feed rate
    elif mFeed:
        tool_feed = float(mFeed.group(1))

    # Spindle speed
    elif mSpeed:
        tool_rpm = float(mSpeed.group(1))

    # GCode
    elif mgCode:
        move_cut = int(mMoveCut.group(1))
        if step_number == 0:
            # Create all of the shapes the first time real GCode is encountered.
            if verbose:
                print('Header has been read')

            # Create a polygon to represent the gear blank (just slightly larger than the finished gear)
            gear_blank = circle(v['outside_diameter'] / 2 + v['module'] / 2)

            # Create a polygon to for the pitch circle
            pitch_circle = circle(v['pitch_diameter'] / 2)

            # Create a polygon to for the dedendum circle
            dedendum_circle = circle(v['pitch_diameter'] / 2 - v['h_addendum'])

            # Create a polygon to for the clearance circle
            clearance_circle = circle(v['pitch_diameter'] / 2 - v['h_dedendum'])

            # Create a polygon to represent the cutting tool
            direction = 1 if right_rotary else -1
            half_tip = tool_tip_height / 2.
            y = half_tip + tan(radians(tool_angle / 2.)) * tool_depth
            shaft = tool_radius - tool_depth
            print('Tool: shaft_diameter=%9.4fmm 2*y=%9.4fmm' % (shaft*2, y))
            print('Tool: shaft_diameter=%9.4f 64ths 2*y=%9.4f 64ths' % (shaft*2/25.4*64, y/25.4*64))
            cutter_shaft = Polygon([
                (shaft, v['outside_radius']),
                (shaft, y),
                (shaft, -y-tool_shaft_extension),
                (-shaft, -y-tool_shaft_extension),
                (-shaft, y),
                (-shaft, v['outside_radius']),
                ])
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
            cutter_shaft = Polygon(tool.shaft_poly(v['outside_radius']))
            cutter = Polygon(tool.cutter_poly(v['outside_radius']))
            extra = 0.2
            extra_cutter = Polygon([
                (shaft, v['outside_radius']),
                (shaft, y),
                (tool_radius+extra, half_tip),
                (tool_radius+extra, -half_tip),
                (shaft, -y),
                (-shaft, -y),
                (-tool_radius-extra, -half_tip),
                (-tool_radius-extra, half_tip),
                (-shaft, y),
                (-shaft, v['outside_radius']),
                ])

            # total hack for mercury, since we don't have these params in this program
            # TODO-need half_tooth?  (use new rack code?)
            v['teeth'] = int(v['teeth'])
            gg_rack = Rack(v['module'], degrees(v['pressure_angle']))
            rack_polygon = Polygon(gg_rack.points(v['teeth']))
            gg = gear_plot.GearInvolute(v['teeth'],
                                        module=v['module'], relief_factor=v['relief_factor'],
                                        pressure_angle=degrees(v['pressure_angle']))
            gg_poly = gg.instance().poly
            gg_base_circle = circle(gg.base_radius)
            if cycloidal_target := v.get('cycloidal_target'):
                cp = CycloidalPair(v['wheel_teeth'], v['pinion_teeth'], v['module'])
                if cycloidal_target == 'wheel':
                    gg_poly = cp.wheel().poly
                else:
                    gg_poly = cp.pinion().poly

            if sa:
                if zoom:
                    # cx = v['outside_radius']
                    cx = v['pitch_diameter'] / 2
                    cy = 0
                    # zr = max(v['h_total'] * 3, v['z_max'])
                    if zoom == 1:
                        zr = v['h_total'] * 2
                    else:
                        zr = v['h_total'] * 0.9
                else:
                    cx, cy = 0, 0
                    zr = v['pitch_diameter']
                sa.model_bbox = BBox(cx - zr, cy - zr, cx + zr, cy + zr)


        # Move and cut based on each axis
        for axis, amt in mgCode:
            step_number += 1
            amt = float(amt)
            if axis == 'A':
                if abs(cur_angle - amt) > 20:
                    print('WARNING: Large Angle move: %.4f -> %.4f @ %d %s' % (cur_angle, amt, line_number+1, l))
                gear_blank = rotate(gear_blank, cur_angle - amt, origin=(0, 0))
                cur_angle = amt
            elif axis == 'X':
                if cutter_y:
                    cur_cutter = translate(cutter, cutter_y, cutter_z)
                    cur_shaft: Polygon = translate(cutter_shaft, cutter_y, cutter_z)
                    area_start = gear_blank.area
                    if not gear_blank.is_valid:
                        gear_blank = gear_blank.buffer(0)
                    # TODO-even better would be to define a shaft-no-go-zone and use that
                    #     -probably just original gear blank plus a little bit would be good
                    intersection = cur_shaft.intersection(gear_blank)
                    if not hasattr(intersection, 'exterior'):
                        print('ERROR: Shaft intersection has no exterior', line_number+1, l)
                    elif cur_shaft.intersection(gear_blank).exterior:
                        print('ERROR: Shaft intersects gear blank', line_number+1, l)
                    gear_blank = gear_blank.difference(cur_cutter)

                    # Deal with an acute cutter trimming off a shard
                    if type(gear_blank) == MultiPolygon:
                        log_cut("Sharding:", line_number+1, l)
                        big_poly, area = None, 0.
                        for polygon in gear_blank:
                            if polygon.area > area:
                                big_poly, area = polygon, polygon.area
                        gear_blank = big_poly

                    # Track material removal
                    amountCut = area_start - gear_blank.area
                    if amountCut > 0.:
                        cuttings.append(amountCut)
                        if move_cut == 0:
                            log_cut("Error:", line_number+1, l)
                    elif move_cut == 1:
                        log_cut("Delete:", line_number+1, l)

                    # Write an animation frame
                    if animate and (teeth_to_draw == 'all' or tooth in teeth_to_draw):
                        if tooth != tooth_last:
                            print('Render: tooth=%d' % tooth)
                            tooth_last = tooth
                        show_rotated = True
                        with sa.next_step() as dc:
                            def poly(pp: Polygon, fill=None, outline='black'):
                                # print(pp.exterior.coords)
                                dc.polygon(pp.exterior.coords, fill, outline)
                            if show_rotated:
                                rotated_gear_blank = rotate(gear_blank, cur_angle, origin=(0, 0))
                                poly(rotated_gear_blank, 'blue')
                                rotated_cutter = rotate(cur_cutter, cur_angle, origin=(0, 0))
                                poly(rotated_cutter, 'red')
                                rotated_shaft = rotate(cur_shaft, cur_angle, origin=(0, 0))
                                poly(rotated_shaft, 'yellow')
                                cur_extra = translate(extra_cutter, cutter_y, cutter_z)
                                rotated_extra = rotate(cur_extra, cur_angle, origin=(0, 0))
                                poly(rotated_extra, None, '#FF8080')
                            else:
                                poly(gear_blank, 'blue')
                                poly(cur_cutter, 'red')
                                poly(cur_shaft, 'yellow')
                            poly(pitch_circle, None, 'cyan')
                            poly(clearance_circle, None, 'yellow')
                            poly(gg_base_circle, None, 'brown')
                            if show_rotated:
                                dc.polygon(gg_poly, None, 'green')
                            else:
                                rotated_gg = rotate(Polygon(gg_poly), -cur_angle, origin=(0, 0))
                                poly(rotated_gg, None, 'green')
                            r = v['pitch_diameter'] / 2.
                            ca = radians(cur_angle)
                            rack_shifted = translate(rack_polygon, 0, -ca * r, 0)
                            # poly(rack_shifted, None, 'white')

                        if zoom:
                            clip = box(v['outside_radius']-v['h_total']*3, -v['z_max'], v['outside_radius']+v['module']*3, v['z_max'])
                            try:
                                # plt.plot(*cur_cutter.intersection(clip).exterior.xy, color='r')
                                plt.plot(*pitch_circle.intersection(clip).exterior.xy, color='g')
                                plt.plot(*clearance_circle.intersection(clip).exterior.xy, color='c')
                                if show_rotated:
                                    rotated_gear_blank = rotate(gear_blank, cur_angle, origin=(0, 0))
                                    plt.plot(*rotated_gear_blank.intersection(clip).exterior.xy, color='b')
                                    rotated_cutter = rotate(cur_cutter, cur_angle, origin=(0, 0))
                                    plt.plot(*rotated_cutter.intersection(clip).exterior.xy, color='cyan')
                                    # plt.plot(*cur_cutter.intersection(clip).exterior.xy, color='r')
                                else:
                                    plt.plot(*gear_blank.intersection(clip).exterior.xy, color='b')
                                    plt.plot(*cur_cutter.intersection(clip).exterior.xy, color='r')
                                #plt.plot((0., cos(radians(-cur_angle)) * v['outside_radius']), (0., sin(radians(-cur_angle)) * v['outside_radius']), color='b')
                                #plt.plot((-direction*(v['outside_radius'] - v['h_total']), -direction*(v['outside_radius'] - v['h_total'])), (-v['z_max'], v['z_max']), color='y')
                                plt.grid()
                                plt.axis('equal')
                                camera.snap()
                            except AttributeError:
                                pass
                        else:
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
                cutter_y = amt + yerror
            elif axis == 'Z':
                cutter_z = amt + zerror

# Create the animation
if animate:
    if verbose:
        print('Generating animation "%s"' % animationFile)
    animation = camera.animate(1000)
    animation.save(animationFile, writer='pillow')
    sa.save_animation('/tmp/sa_gears.gif')

# Create a picture picture of the gear
if picture:
    gear_blank = rotate(gear_blank, cur_angle, origin=(0, 0))
    if verbose:
        print('Generating picture "%s"' % pictureFile)
    fig = plt.figure()
    if zoom:
        clip = Polygon([
            (-v['module'] * 2, v['outside_radius']+v['h_addendum']),
            (v['module'] * 2, v['outside_radius']+v['h_addendum']),
            (v['module'] * 2, v['outside_radius'] - v['h_total'] - v['h_addendum']),
            (-v['module'] * 2, v['outside_radius'] - v['h_total'] - v['h_addendum'])])
        plt.plot(*pitch_circle.intersection(clip).exterior.xy, color='g')
        plt.plot(*clearance_circle.intersection(clip).exterior.xy, color='c')
        plt.plot(*dedendum_circle.intersection(clip).exterior.xy, color='m')
        final_blank = gear_blank.intersection(clip)
        if not final_blank.exterior:
            print('ERROR: Gear blank does not intersect with zoom window')
        else:
            plt.plot(*final_blank.exterior.xy, color='b')
    else:
        plt.plot(*pitch_circle.exterior.xy, color='g')
        plt.plot(*clearance_circle.exterior.xy, color='c')
        plt.plot(*dedendum_circle.exterior.xy, color='m')
        plt.plot(*gear_blank.exterior.xy, color='b')
        plt.grid()
    plt.axis('equal')
    plt.savefig(pictureFile)

    if sys.platform == 'darwin' and gear_plot.SHOW_INTERACTIVE:
        # Redo plot and display for interactive viewing and zooming
        plt.plot(*pitch_circle.exterior.xy, color='g')
        plt.plot(*clearance_circle.exterior.xy, color='c')
        plt.plot(*dedendum_circle.exterior.xy, color='m')
        plt.plot(*gear_blank.exterior.xy, color='b')
        plt.plot(*zip(*gg_poly), color='#DDDDDD')
        plt.plot(*zip(*gg.gen_rack_tooth()), 'green')
        gg.plot_show(2)

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
    materialRR = area * tool_feed

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
    print("Cutting rate:")
    print("    Time per each pass: %g mins" % cut_time)
    print("    Cuts per pass: %g" % cut_count)
    print("    Material per flute: %g %s" % (conv3 * materialPerFlute, units3))
    print("    Material removal rate: %g %s/min" % (conv3 * materialRR, units3))
