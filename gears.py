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
from math import sin, cos, tan, atan2, radians, degrees, sqrt, pi, tau
import configargparse

import gear_plot


def rotate(a, x, y):
    """Return point x,y rotated by angle a (in radians)."""
    return x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)


class Tool():
    r"""
        The Tool class holds the specifications of the cutting tool.
          /\
         /  \
         |   |
         |   ---------
         |   ---------
         |   |
         \  /
          \/
    """

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

    def __init__(self, tool, module=1., pressure_angle=20., relief_factor=1.25, steps=5, root_steps=1,
                 cutter_clearance=2., right_rotary=False, algo='old'):
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
        self.algo = algo

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

    def gcode_vars(self, locals_to_dump):
        """Include all of the generating parameters in the G Code header"""

        var_t = ['z_max', 'module', 'teeth', 'blank_thickness', 'tool', 'relief_factor',
                 'pressure_angle', 'steps', 'cutter_clearance', 'right_rotary', 'h_addendum',
                 'h_dedendum', 'h_total', 'circular_pitch', 'pitch_diameter',
                 'outside_diameter', 'outside_radius', 'tool_angle_offset', 'x_start',
                 'x_end']
        gcode = []
        for var in var_t:
            value = locals_to_dump[var] if var in locals_to_dump else getattr(self, var)
            gcode.append('( %17s: %-70s )' % (var, value))
        return '\n'.join(gcode)

    def gcode_guts(self, gg: gear_plot.Gear, cut: 'Cut', teeth_to_make) -> str:
        """Generate the gcode guts"""
        gcode = []
        cuts = gg.cuts_for_mill(degrees(self.tool.angle), self.tool.tip_height)
        # Generate a tooth profile for ever tooth requested
        teeth_to_make = min(5, teeth_to_make)
        for tooth in range(teeth_to_make):
            tooth_angle_offset = 360 * tooth / gg.teeth

            rmin = rmax = cuts[0][0]
            for r, y, z in cuts:
                rmin = min(rmin, r)
                rmax = max(rmax, r)

            gcode.append('')
            gcode.append("( Tooth: %d  rot:%.2f->%.2f)" % (tooth, rmin, rmax))

            for idx, (rotation, y, z) in enumerate(cuts):
                rotation += tooth_angle_offset
                y += self.tool.radius
                # print('G? A%.5f Y%.5f Z%.5f' % (rotation, y, z))
                gcc = cut.cut(-rotation, y, z)
                # print(gcc)
                gcode.append(gcc)

        return '\n'.join(gcode)

    def generate_new(self, teeth, blank_thickness, teeth_to_make=0):
        """Generate the gcode for cutting a gear."""

        if teeth <= 0:
            raise ValueError('Gear: Number of teeth must be greater than 0.')
        if blank_thickness <= 0:
            raise ValueError('Gear: Blank thickness must be greater than 0.')

        gg = gear_plot.Gear(teeth,
                            module=self.module, relief_factor=self.relief_factor,
                            pressure_angle=degrees(self.pressure_angle))
        if gear_plot.SHOW_INTERACTIVE:
            gg.plot('red', tool_angle=self.tool.angle)
            gg.plot_show(2)

        # Calculate the variables used to generate the gear teeth
        h_addendum = self.module
        h_dedendum = self.module * self.relief_factor
        h_total = h_addendum + h_dedendum
        circular_pitch = self.module * pi
        pitch_diameter = self.module * teeth
        pitch_radius = pitch_diameter / 2.
        outside_diameter = pitch_diameter + 2 * h_addendum
        outside_radius = outside_diameter / 2.

        half_tooth = circular_pitch / 4.
        half_tool_tip = self.tool.tip_height / 2.
        tip_offset_y = h_dedendum
        tip_offset_z = half_tool_tip + tan(self.tool.angle / 2) * h_dedendum
        tool_angle_offset = self.tool.angle / 2. - self.pressure_angle

        z_offset = (circular_pitch / 2. - 2. * sin(self.pressure_angle) * h_dedendum - self.tool.tip_height) / 2.
        root_incr = z_offset / (self.root_steps + 1)

        x_offset = self.cutter_clearance + blank_thickness / 2. + sqrt(
            self.tool.radius ** 2 - (self.tool.radius - h_total) ** 2)
        mill = self.tool.mill
        angle_direction = 1 if self.right_rotary else -1
        x_start, x_end = -angle_direction * x_offset, angle_direction * x_offset

        # Determine the maximum amount of height (or depth) in the Z axis before the tip
        # of the cutter won't intersect with the gear blank.
        # z_max = sqrt(outside_radius**2 - (outside_radius - h_total)**2) + self.tool.tip_height
        z_max = outside_radius
        z_incr = z_max / (self.steps * 2)

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

        gcode = [self.gcode_vars(locals())]

        # Move to safe initial position
        cut = Cut(mill, x_start, x_end, -angle_direction * self.cutter_clearance,
                  -angle_direction * (outside_radius + self.tool.radius + self.cutter_clearance), outside_radius)
        gcode.append(cut.start())

        gcode.append(self.gcode_guts(gg, cut, teeth_to_make))

        return '\n'.join(gcode)

    def generate_old(self, teeth, blank_thickness, teeth_to_make=0):
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

        half_tooth = circular_pitch / 4.
        half_tool_tip = self.tool.tip_height / 2.
        tip_offset_y = h_dedendum
        tip_offset_z = half_tool_tip + tan(self.tool.angle / 2) * h_dedendum
        tool_angle_offset = self.tool.angle / 2. - self.pressure_angle

        z_offset = (circular_pitch / 2. - 2. * sin(self.pressure_angle) * h_dedendum - self.tool.tip_height) / 2.
        root_incr = z_offset / (self.root_steps + 1)

        x_offset = self.cutter_clearance + blank_thickness / 2. + sqrt(self.tool.radius ** 2 - (self.tool.radius - h_total) ** 2)
        mill = self.tool.mill
        angle_direction = 1 if self.right_rotary else -1
        x_start, x_end = -angle_direction * x_offset, angle_direction * x_offset

        # Determine the maximum amount of height (or depth) in the Z axis before the tip
        # of the cutter won't intersect with the gear blank.
        # z_max = sqrt(outside_radius**2 - (outside_radius - h_total)**2) + self.tool.tip_height
        z_max = outside_radius
        z_incr = z_max / (self.steps * 2)

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

        gcode = [self.gcode_vars(locals())]

        # Move to safe initial position
        cut = Cut(mill, x_start, x_end, -angle_direction * self.cutter_clearance,
                  -angle_direction * (outside_radius + self.tool.radius + self.cutter_clearance), outside_radius)
        gcode.append(cut.start())

        # Generate a tooth profile for ever tooth requested
        for tooth in range(teeth_to_make):
            tooth_angle_offset = 2. * pi * tooth / teeth
            gcode.append('')
            gcode.append("( Tooth: %d)" % tooth)

            # For testing, set this to limit the steps and eliminate center clearing
            # debug_cuts = [-2]
            debug_cuts = None

            # The shape of the tooth (actually the space removed to make the tooth)
            # is created iteratively with a number of steps. More steps means greater
            # accuracy but longer run time.

            # Bottom of the tooth (top of the slot)
            #for z_step in range(self.steps+1, -1, -1):
            for z_step in debug_cuts or range(-self.steps, self.steps+1):
                # center of the cutting tooth
                z = z_step * z_incr
                y = pitch_radius

                # blank angle of the center point of the "cutting tooth"
                angle = z / y

                # move z to tooth edge
                z += half_tooth

                # Find the rotated pitch radius and z of the actual cutter
                y_point, z_point = rotate(tool_angle_offset, y, z)
                angle += tool_angle_offset

                # Find the tip of the actual cutter
                y_point -= tip_offset_y
                z_point -= tip_offset_z

                # cut
                if sqrt(y_point**2 + z_point**2) < outside_radius:
                    gcode.append(cut.cut(
                        angle_direction * degrees(tooth_angle_offset + angle),
                        -angle_direction * (self.tool.radius + y_point),
                        z_point))

            # Top of the tooth (bottom of the slot)
            for z_step in debug_cuts or range(-self.steps, self.steps+1):
                # height of the center of the cutting tooth at the pitch radius
                z = -z_step * z_incr
                y = pitch_radius

                # blank angle of the center point of the "cutting tooth"
                angle = z / y

                # move z to pressure_angle edge
                z -= half_tooth

                # Find the rotated pitch radius and z of the actual cutter
                y_point, z_point = rotate(-tool_angle_offset, y, z)
                angle -= tool_angle_offset

                # Find the tip of the actual cutter
                y_point -= tip_offset_y
                z_point += tip_offset_z

                # cut
                if sqrt(y_point**2 + z_point**2) < outside_radius:
                    gcode.append(cut.cut(
                        angle_direction * degrees(tooth_angle_offset + angle),
                        -angle_direction * (self.tool.radius + y_point),
                        z_point))

            # Center of the slot
            for z_step in range(-self.root_steps, self.root_steps+1):
                if debug_cuts:
                    break
                z = z_step * root_incr
                angle = z / (pitch_radius - h_dedendum)
                gcode.append(cut.cut(
                    (angle_direction * degrees(angle + tooth_angle_offset)),
                    (-angle_direction * (self.tool.radius + pitch_radius - h_dedendum)),
                    0))
            
        return '\n'.join(gcode)

    def generate_alignment_cuts(self, teeth, blank_thickness, teeth_to_make=0):
        """Generate the gcode for creating alignment cuts."""

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

        half_tooth = circular_pitch / 4.
        half_tool_tip = self.tool.tip_height / 2.
        tip_offset_y = h_dedendum
        tip_offset_z = half_tool_tip + tan(self.tool.angle / 2) * h_dedendum
        tool_angle_offset = self.tool.angle / 2. - self.pressure_angle

        z_offset = (circular_pitch / 2. - 2. * sin(self.pressure_angle) * h_dedendum - self.tool.tip_height) / 2.
        root_incr = z_offset / (self.root_steps + 1)

        x_offset = self.cutter_clearance + blank_thickness / 2. + sqrt(self.tool.radius ** 2 - (self.tool.radius - h_total) ** 2)
        mill = self.tool.mill
        angle_direction = 1 if self.right_rotary else -1
        x_start, x_end = -angle_direction * x_offset, angle_direction * x_offset

        # Determine the maximum amount of height (or depth) in the Z axis before the tip
        # of the cutter won't intersect with the gear blank.
        # z_max = sqrt(outside_radius**2 - (outside_radius - h_total)**2) + self.tool.tip_height
        z_max = outside_radius
        z_incr = z_max / (self.steps * 2)

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

        gcode = [self.gcode_vars(locals())]

        # Move to safe initial position
        cut = Cut(mill, x_start, x_end, -angle_direction * self.cutter_clearance,
                  -angle_direction * (outside_radius + self.tool.radius + self.cutter_clearance), outside_radius)
        gcode.append(cut.start())

        # Generate a tooth profile for ever tooth requested
        for tooth in range(teeth_to_make):
            tooth_angle_offset = tau * tooth / teeth
            gcode.append('')
            gcode.append("( Tooth: %d)" % tooth)

            # Make two small cuts at 90 degrees to each other to
            # verify alignment of tools
            for align_dir in [1, -1]:
                angle = align_dir * (self.tool.angle / 2 - radians(45))

                y_point, z_point = rotate(angle, pitch_radius, 0)
                z_point -= align_dir * half_tool_tip
                gcode.append(cut.cut(
                    angle_direction * degrees(angle + tooth_angle_offset),
                    -angle_direction * (self.tool.radius + y_point),
                    z_point))

        return '\n'.join(gcode)

    def generate(self, teeth, blank_thickness, teeth_to_make=0) -> str:
        if self.algo == 'new':
            print('Generate: algo=%s -> new' % self.algo)
            return self.generate_new(teeth, blank_thickness, teeth_to_make)
        elif self.algo == 'align':
            print('Generate: algo=%s -> alignment' % self.algo)
            return self.generate_alignment_cuts(teeth, blank_thickness, teeth_to_make)
        else:
            print('Generate: algo=%s -> old' % self.algo)
            return self.generate_old(teeth, blank_thickness, teeth_to_make)


class Cut():
    """Cut is used to generate the gcode for the actual cutting and retraction
    of the tool and the gear blank.  This can get a bit complicated due to the
    different styles of cutting (climb, conventional, both) and the various
    setups of the rotary table on the left or right side.
    """

    def __init__(self, mill, x_start, x_end, y_backoff, y_backoff_full, outside_radius):
        self.mill = mill
        self.x_start = x_start
        self.x_end = x_end
        self.y_backoff = y_backoff
        self.stroke = 0
        self.y_backoff_full = y_backoff_full
        self.outside_radius = outside_radius

    def start(self) -> str:
        """Return the starting gcode."""
        starting_x = self.x_end if self.mill == 'climb' else self.x_start
        gcode = [
            '',
            'G30',
            'G0 Y%g' % self.y_backoff_full,
            'G0 X%.4f' % starting_x,
            'G0 Z%g' % self.outside_radius
        ]
        return '\n'.join(gcode)

    def cut(self, a, y, z):
        """Create gcode for the cut/return stroke."""
        # This code is very cautious and only makes A or Z moves if Y is at fully safe point
        if self.mill == 'climb':
            ret = ["G1 X%.4f" % self.x_start,
                   "G0 Y%.4f" % self.y_backoff_full,
                   "G0 X%.4f" % self.x_end]
        elif self.mill == 'conventional':
            ret = ["G1 X%.4f" % self.x_end,
                   "G0 Y%.4f" % self.y_backoff_full,
                   "G0 X%.4f" % self.x_start]
        else:
            assert "This has not been tested" == "yes, it has not"
            ret = ["G1 X%.4f" % [self.x_start, self.x_end][self.stroke],
                   "G0 Y%.4f" % self.y_backoff_full]
            self.stroke = (self.stroke + 1) % 2

        align_for_cut = ["G0 A%.4f Z%.4f" % (a, z), "G0 Y%.4f" % y]
        return '\n'.join(align_for_cut + ret)


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
    p.add('--algo', type=str, default='old', help='Which algorithm to use for gcode generation')

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

    # Alignment test
    p.add('--align', type=bool, default=False, help='Generate alignment cuts')

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
                    cutter_clearance=args.clear, right_rotary=args.right,
                    algo='align' if args.align else args.algo)

        print(gear.header(), file=out)
        print(gear.generate(args.teeth, args.thick, args.make), file=out)
        print(gear.footer(), file=out)
    except ValueError as error:
        print(error, file=sys.stderr)


if __name__ == '__main__':
    main()
