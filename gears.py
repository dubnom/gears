#!/usr/bin/python3

"""
G Code generator for cutting metric involute gears.

Supported tools:
    double angle shaft cutters
    double bevel cutters
    slitting saws

All units are specified in millimeters or degrees.

Copyright 2020 - Michael Dubno - New York
"""

# FIX: Add multiple passes
# FIX: Add support for DP

import sys
from math import sin, cos, atan2, radians, degrees, sqrt, pi
import configargparse


def rotate(a, x, y):
    """Return point x,y rotated by angle a (in radians)."""
    return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)


class Tool():
    """The Tool class holds the specifications of the cutting tool."""

    def __init__(self, angle=40., depth=3., radius=10., tip_height=0.,
                 number=1, rpm=2000, feed=200, flutes=4, mist=False, flood=False,
                 mill='both'):
        if angle < 0.:
            raise ValueError('Tool: Angle must be greater than or equal to 0')
        if depth <= 0.:
            raise ValueError('Tool: Depth must be greater than 0')
        if radius < depth:
            raise ValueError('Tool: Radius must be greater than depth')
        if tip_height < 0:
            raise ValueError('Tool: tip_height must be greater than or equal to 0')
        if mill not in ['both', 'climb', 'conventional']:
            raise ValueError('Tool: mill must be "both", "climb", or "conventional')

        self.angle = radians(angle)
        self.depth = depth
        self.radius = radius
        self.tip_height = tip_height
        self.number = number
        self.rpm = rpm
        self.feed = feed
        self.flutes = flutes
        self.mist = mist
        self.flood = flood
        self.mill = mill

    def __str__(self):
        return "(Angle: {}, Depth: {}, Radius: {}, TipHeight: {}, Flutes: {})".format(
            degrees(self.angle), self.depth, self.radius, self.tip_height, self.flutes)


class Gear():
    """The Gear class is used to generate G Code of involute gears."""

    def __init__(self, tool, module=1., pressure_angle=20., relief_factor=1.25,
                 steps=5, root_steps=1, cutter_clearance=2., right_rotary=False):
        if module <= 0:
            raise ValueError('Gear: Module must be greater than 0.')
        if pressure_angle <= 0:
            raise ValueError('Gear: Pressure angle must be greater than 0.')
        if steps < 0:
            raise ValueError('Gear: Steps must be greater than or equal to 0.')

        self.module = module
        self.tool = tool
        self.relief_factor = relief_factor
        self.pressure_angle = radians(pressure_angle)
        self.steps = steps
        self.root_steps = root_steps
        self.cutter_clearance = cutter_clearance
        self.right_rotary = right_rotary

    def header(self):
        """Return the gcode for the top of the file."""

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
        """Return the gcode for the bottom of the file."""

        return \
"""\
M5 M9
G30
M30
%"""

    def generate(self, teeth, blank_thickness, teeth_to_make=0):
        """Generate the gcode for cutting a gear."""

        if teeth <= 0:
            raise ValueError('Gear: Number of teeth must be greater than 0.')
        if blank_thickness <= 0:
            raise ValueError('Gear: Blank thickness must be greater than 0.')

        # Calculate the variables used to generate the gear teeth
        h_addendum = self.module
        h_dedendum = self.module * self.relief_factor
        h_total = h_addendum + h_dedendum
        circular_pitch = self.module * pi
        pitch_diameter = self.module * teeth
        pitch_radius = pitch_diameter / 2.
        outside_diameter = pitch_diameter + 2 * h_addendum
        outside_radius = outside_diameter / 2.

        tool_angle_offset = self.tool.angle / 2. - self.pressure_angle
        z_offset = (circular_pitch / 2. - 2. * sin(self.pressure_angle) * h_dedendum - self.tool.tip_height) / 2.
        root_incr = z_offset / (self.root_steps + 1)
        print(z_offset, root_incr, self.root_steps)

        x_offset = self.cutter_clearance + blank_thickness / 2. + sqrt(self.tool.radius ** 2 - (self.tool.radius - h_total) ** 2)
        mill = self.tool.mill
        angle_direction = 1 if self.right_rotary else -1
        x_start, x_end = -angle_direction * x_offset, angle_direction * x_offset

        # Determine the maximum amount of height (or depth) in the Z axis before part of the cutter
        # won't intersect with the gear blank.
        z_max = sqrt(outside_radius**2 - (outside_radius - h_total)**2)
        z_incr = z_max / (self.steps + 1)

        # A partial number of teeth can be created if "teeth_to_make" is set,
        # otherwise all of the gears teeth are cut.
        if teeth_to_make == 0 or teeth_to_make > teeth:
            teeth_to_make = teeth

        # Make sure the cutter is big enough
        if h_total > self.tool.depth:
            raise ValueError("Cutter depth is too shallow for tooth height")

        # Make sure the cutter shaft doesn't hit the gear blank.
        shaft_radius = self.tool.radius - self.tool.depth
        y_point, z_point = pitch_radius, z_max
        y_tool, z_tool = rotate(-tool_angle_offset, y_point, z_point)
        y = self.tool.radius + y_tool - h_dedendum
        shaft_clearance = y - outside_radius - shaft_radius
        if shaft_clearance < 0:
            raise ValueError("Cutter shaft hits gear blank by %g mm" % -shaft_clearance)

        # Include all of the generating parameters in the G Code header
        var_t = ['z_max', 'module', 'teeth', 'blank_thickness', 'tool', 'relief_factor',
                 'pressure_angle', 'steps', 'cutter_clearance', 'right_rotary', 'h_addendum',
                 'h_dedendum', 'h_total', 'circular_pitch', 'pitch_diameter',
                 'outside_diameter', 'outside_radius', 'tool_angle_offset', 'x_start',
                 'x_end']
        gcode = []
        for var in var_t:
            if var in locals():
                gcode.append('( %17s: %-70s )' % (var, locals()[var]))
            else:
                gcode.append('( %17s: %-70s )' % (var, getattr(self, var)))

        # Move to safe initial position
        cut = Cut(mill, x_start, x_end, -angle_direction * self.cutter_clearance)
        gcode.append('')
        gcode.append('G30')
        gcode.append(cut.start())
        gcode.append('G0 Y%g' % (-angle_direction * (outside_radius + self.tool.radius + self.cutter_clearance)))
        gcode.append('G0 Z%g' % outside_radius)

        # Generate a tooth profile for ever tooth requested
        for tooth in range(teeth_to_make):
            tooth_angle_offset = 2. * pi * tooth / teeth
            gcode.append('')
            gcode.append("( Tooth: %d)" % tooth)

            # The shape of the tooth (actually the space removed to make the tooth)
            # is created iteratively with a number of steps. More steps means greater
            # accuracy but longer run time.
            half_tooth = circular_pitch / 4.

            # Bottom of the tooth (top of the slot)
            for z_step in range(self.steps+1, -1, -1):
                # height of the center of the cutting tooth at the pitch radius
                z = -z_step * z_incr

                # blank angle of the center point of the "cutting tooth"
                angle = atan2(z, pitch_radius)

                # move z to tooth edge
                z += half_tooth

                # Find the tip of the actual cutter (and new angle)
                y_point, z_point = rotate(tool_angle_offset, pitch_radius, z)
                angle += tool_angle_offset
                y_point -= cos(self.tool.angle / 2) * h_dedendum
                z_point -= sin(self.tool.angle / 2) * h_dedendum

                # cut
                gcode.append(cut.cut(
                    angle_direction * degrees(tooth_angle_offset + angle),
                    -angle_direction * (self.tool.radius + y_point),
                    z_point - self.tool.tip_height / 2.))

            # Top of the tooth (bottom of the slot)
            for z_step in range(self.steps, -1, -1):
                # height of the center of the cutting tooth at the pitch radius
                z = z_step * z_incr

                # blank angle of the center point of the "cutting tooth"
                angle = atan2(z, pitch_radius)

                # move z to pressure_angle edge
                z -= half_tooth

                # Find the tip of the actual cutter (and new angle)
                y_point, z_point = rotate(-tool_angle_offset, pitch_radius, z)
                angle -= tool_angle_offset
                y_point -= cos(self.tool.angle / 2) * h_dedendum
                z_point += sin(self.tool.angle / 2) * h_dedendum

                # cut
                gcode.append(cut.cut(
                    angle_direction * degrees(tooth_angle_offset + angle),
                    -angle_direction * (self.tool.radius + y_point),
                    z_point + self.tool.tip_height / 2.))
            # Center of the slot
            #for z_step in range(-self.root_steps, self.root_steps+1):
            #    z = z_step * root_incr
            #    angle = z / (pitch_radius - h_dedendum)
            #    gcode.append(cut.cut(
            #        (angle_direction * degrees(angle + tooth_angle_offset)),
            #        (-angle_direction * (self.tool.radius + pitch_radius - h_dedendum)),
            #        0))
            
            #for z_step in range(-self.root_steps, self.root_steps+1):
            #    z = z_step * root_incr
            #    angle = z / pitch_radius
            #    gcode.append(cut.cut(
            #        (angle_direction * degrees(tooth_angle_offset + angle)),
            #        (-angle_direction * (pitch_radius + self.tool.radius - h_dedendum)),
            #        0))

        return '\n'.join(gcode)


class Cut():
    """Cut is used to generate the gcode for the actual cutting and retraction
    of the tool and the gear blank.  This can get a bit complicated due to the
    different styles of cutting (climb, conventional, both) and the various
    setups of the rotary table on the left or right side.
    """

    def __init__(self, mill, x_start, x_end, y_backoff):
        self.mill = mill
        self.x_start = x_start
        self.x_end = x_end
        self.y_backoff = y_backoff
        self.stroke = 0

    def start(self):
        """Return the starting gcode."""
        if self.mill == 'climb':
            return "G0 X%g" % self.x_end
        return "G0 X%g" % self.x_start

    def cut(self, a, y, z):
        """Create gcode for the cut/return stroke."""
        if self.mill == 'climb':
            ret = ["G1 X%g" % self.x_start,
                   "G0 Y%g" % (y + self.y_backoff),
                   "G0 X%g" % self.x_end,
                   "G0 Y%g" % y]
        elif self.mill == 'conventional':
            ret = ["G1 X%g" % self.x_end,
                   "G0 Y%g" % (y + self.y_backoff),
                   "G0 X%g" % self.x_start,
                   "G0 Y%g" % y]
        else:
            ret = ["G1 X%g" % [self.x_start, self.x_end][self.stroke]]
            self.stroke = (self.stroke + 1) % 2

        return '\n'.join(["G0 A%g Y%g Z%g" % (a, y, z)] + ret)


def main():
    """Parse the command line and generate gears."""

    p = configargparse.ArgParser(
        default_config_files=['gears.cfg'],
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        description="Generate G Code to create involute spur gears.",
        epilog="""
            By using simple cutters like double angle shaft cutters or #8 gear cutter, you can cut
            most modules, pressure angles, and tooth count involute spur gears.  The limiting factors
            are the size of the cutter and the working envelope of the mill.

            A rotary 4th-axis is used to rotate the gear blank with the cutting tool held in the spindle.
                """)
    p.add('out', nargs='?', type=configargparse.FileType('w'), default=sys.stdout)
    p.add('--config', '-X', is_config_file=True, help='Config file path')

    # Tool arguments
    p.add('--tool', '-T', is_config_file=True, help='Tool config file')
    p.add('--angle', '-A', type=float, default=40., help='Tool: included angle in degrees')
    p.add('--depth', '-D', type=float, default=5., help='Tool: depth of cutting head in mm')
    p.add('--height', '-H', type=float, default=0., help='Tool: distance between the top and bottom of cutter at tip in mm')
    p.add('--diameter', '-I', type=float, default=15., help='Tool: cutting diameter at tip in mm')
    p.add('--number', '-N', type=int, default=1, help='Tool: tool number')
    p.add('--rpm', '-R', type=float, default=2000., help='Tool: spindle speed')
    p.add('--feed', '-F', type=float, default=200., help='Tool: feed rate')
    p.add('--flutes', '-U', type=int, default=4, help='Tool: flutes')
    p.add('--mist', '-M', action='store_true', help='Tool: turn on mist coolant')
    p.add('--flood', '-L', action='store_true', help='Tool: turn on flood coolant')
    p.add('--mill', default='conventional', choices=['both', 'climb', 'conventional'], help='Tool: cutting method')

    # Gear type arguments
    p.add('--gear', '-g', is_config_file=True, help='Gear config file')
    p.add('--module', '-m', type=float, default=1., help='Module of the gear')
    p.add('--pressure', '-p', type=float, default=20., help='Pressure angle in degrees')
    p.add('--relief', type=float, default=1.25, help='Relief factor (for the dedendum)')
    p.add('--steps', '-s', type=int, default=5, help='Steps/tooth face')
    p.add('--roots', '-o', type=int, default=1, help='Number of passes to clean up the root')
    p.add('--clear', '-c', type=float, default=2., help='Cutter clearance from gear blank in mm')
    p.add('--right', '-r', action='store_true', help='Rotary axis is on the right side of the machine')

    # Specific gear arguments
    p.add('--teeth', '-t', type=int, required=True, help='Number of teeth for the entire gear')
    p.add('--thick', '-k', type=float, required=True, help='Thickness of gear blank in mm')
    p.add('--make', type=int, default=0, help='Actual number of teeth to cut.')

    args = p.parse_args()
    out = args.out

    try:
        tool = Tool(angle=args.angle, depth=args.depth, tip_height=args.height,
                    radius=args.diameter / 2., number=args.number, rpm=args.rpm,
                    feed=args.feed, flutes=args.flutes, mist=args.mist,
                    flood=args.flood, mill=args.mill)

        gear = Gear(tool, module=args.module, pressure_angle=args.pressure,
                    relief_factor=args.relief, steps=args.steps, root_steps=args.roots,
                    cutter_clearance=args.clear, right_rotary=args.right)

        print(gear.header(), file=out)
        print(gear.generate(args.teeth, args.thick, args.make), file=out)
        print(gear.footer(), file=out)
    except ValueError as error:
        print(error, file=sys.stderr)


if __name__ == '__main__':
    main()
