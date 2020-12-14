import json
import sys
from math import radians, degrees, tan, cos, sin, pi

import configargparse

from anim.geom import Point, Vector
from gear_base import arc


class Tool:
    r"""
        The Tool class holds the specifications of the cutting tool, shown here
        on its side::

               _
              / \               d_e
             /   \             c   f
             |    |            |   |
            |+    ---------   ab   g---------h
            |                 |
            |+    ---------   AB   G---------h
             |    |            |   |
             \   /             C   F
              \_/               D_E

        * angle is angle(dc,ef)
        * depth is height(cd)
        * radius is height(bd)
        * tip_height is length(de)
        * tip_radius is radius at corners cde & def.  Must be <= tip_height/2
        * shaft_extension is length(ab)
        * shaft_length is length(gh)  [argument to cutter_poly() and shaft_poly()]
        * shaft radius is length(aA)/2 and is computed as radius-depth
        * length(bc) must be zero in the current model
    """

    def __init__(self, angle=40., depth=3., radius=10., tip_height=0., tip_radius=0.,
                 shaft_extension=6.35,
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
        if tip_radius > tip_height / 2:
            raise ValueError('Tool: tip_radius must be less than or equal to tip_height / 2')
        if mill not in ['both', 'climb', 'conventional']:
            raise ValueError('Tool: mill must be "both", "climb", or "conventional')

        self.angle = radians(angle)
        self.depth = depth
        self.radius = radius
        self.tip_height = tip_height
        self.tip_radius = tip_radius
        self.shaft_extension = shaft_extension
        self.number = number
        self.rpm = rpm
        self.feed = feed
        self.flutes = flutes
        self.mist = mist
        self.flood = flood
        self.mill = mill

    @property
    def angle_radians(self):
        return self.angle

    @property
    def angle_degrees(self):
        return degrees(self.angle)

    def __str__(self):
        return "(Angle: {}, Depth: {}, Radius: {}, TipHeight: {}, Extension: {}, Flutes: {})".format(
            self.angle_degrees, self.depth, self.radius, self.tip_height, self.shaft_extension, self.flutes)

    def __repr__(self):
        fields = 'depth,radius,tip_height,tip_radius,shaft_extension,number,rpm,feed,flutes,mist,flood,mill'
        field_vals = ', '.join('%s=%s' % (f, getattr(self, f)) for f in fields.split(','))
        return 'Tool(angle=%s, %s)' % (self.angle_degrees, field_vals)

    def __eq__(self, other):
        return type(self) == type(other) and \
               self.__dict__ == other.__dict__

    @staticmethod
    def add_config_args(p: configargparse.ArgumentParser):
        """Add arguments for tool description"""
        # Tool arguments
        p.add_argument('--tool', '-T', is_config_file=True, help='Tool config file')
        p.add_argument('--angle', '-A', type=float, default=40., help='Tool: included angle in degrees')
        p.add_argument('--depth', '-D', type=float, default=5., help='Tool: depth of cutting head in mm')
        p.add_argument('--height', '-H', type=float, default=0., help='Tool: distance between the top and bottom of cutter at tip in mm')
        p.add_argument('--diameter', '-I', type=float, default=15., help='Tool: cutting diameter at tip in mm')
        p.add_argument('--tip_radius', type=float, default=0., help='Tool: bevel(?) radius at cutter tip in mm')
        p.add_argument('--shaft_extension', type=float, default=6.35, help='Tool: shaft extension beyond bottom of cutter in mm')
        p.add_argument('--number', '-N', type=int, default=1, help='Tool: tool number')
        p.add_argument('--rpm', '-R', type=float, default=2000., help='Tool: spindle speed')
        p.add_argument('--feed', '-F', type=float, default=200., help='Tool: feed rate')
        p.add_argument('--flutes', '-U', type=int, default=4, help='Tool: flutes')
        p.add_argument('--mist', '-M', action='store_true', help='Tool: turn on mist coolant')
        p.add_argument('--flood', '-L', action='store_true', help='Tool: turn on flood coolant')
        p.add_argument('--mill', default='conventional', choices=['both', 'climb', 'conventional'], help='Tool: cutting method')

    @staticmethod
    def from_config_args(args: configargparse.Namespace):
        return Tool(angle=args.angle, depth=args.depth, tip_height=args.height, tip_radius=args.tip_radius,
                    shaft_extension=args.shaft_extension,
                    radius=args.diameter / 2., number=args.number, rpm=args.rpm,
                    feed=args.feed, flutes=args.flutes, mist=args.mist,
                    flood=args.flood, mill=args.mill)

    @staticmethod
    def from_config_file(filename):
        """Load a tool from a config file"""
        parser = configargparse.ArgParser()
        Tool.add_config_args(parser)
        return Tool.from_config_args(parser.parse_args(['--tool=%s' % filename]))

    def to_json(self, indent=None):
        d = dict(**self.__dict__)
        d['angle'] = self.angle_degrees
        return json.dumps(d, sort_keys=True, indent=indent)

    @staticmethod
    def from_dict(d: dict):
        return Tool(**d)

    def cutter_poly(self, shaft_length=40.0):
        """Return a polygon representing this tool"""

        half_tip = self.tip_height / 2.
        tip_y = half_tip + tan(self.angle_radians / 2.) * self.depth
        shaft = self.radius - self.depth

        shaft_top = tip_y + shaft_length
        # TODO-leave shaft out of cutter?
        if self.tip_radius:
            # Where is center for tip radius?
            cx = self.radius-self.tip_radius        # Back up by tip_radius
            cy = half_tip-self.tip_radius*tan(pi/4-self.angle_radians/4)
            center = Point(cx, cy)
            sa = pi/2-self.angle_radians/2
            ep = center + self.tip_radius * Vector(cos(sa), sin(sa))
            t1 = arc(self.tip_radius, 90-self.angle_degrees/2, 0, c=Point(cx, cy), steps=3)
            t2 = [(x, -y) for x, y in reversed(t1)]
            tip = t1 + [(self.radius, 0)] + t2 + []
        else:
            tip = [(self.radius, half_tip), (self.radius, -half_tip)]
        cutter_half = [
            (shaft, shaft_top),
            (shaft, tip_y), ] + tip + [
            (shaft, -tip_y),
            (shaft, -tip_y - self.shaft_extension)
        ]
        return cutter_half + [(-x, y) for x, y in reversed(cutter_half)]

    def shaft_poly(self, shaft_length=40.0):
        """Return a polygon representing this tool"""

        half_tip = self.tip_height / 2.
        tip_y = half_tip + tan(self.angle_radians / 2.) * self.depth
        shaft = self.radius - self.depth

        shaft_top = tip_y + shaft_length
        shaft = [
            (shaft, shaft_top),
            (shaft, tip_y),
            (shaft, -tip_y - self.shaft_extension),
            (-shaft, -tip_y - self.shaft_extension),
            (-shaft, tip_y),
            (-shaft, shaft_top),
        ]
        return shaft

    def plot(self, title='', do_show=True):
        # noinspection PyPackageRequirements
        import matplotlib.pyplot as plt
        plt.plot(*zip(*self.cutter_poly(self.radius)))
        plt.plot(*zip(*self.shaft_poly(self.radius)))
        plt.axis('equal')
        if title:
            plt.title(title)
        if do_show:
            plt.show()


def test(args):
    args = args or ['gear_cutter_06.cfg']
    args = args or ['saw32nd.cfg', 'cutter45.cfg', 'gear_cutter_06.cfg']
    for fn in args:
        t = Tool.from_config_file(fn)
        print(fn)
        print('   ', t)
        print('   ', t.to_json())
        tt = Tool.from_dict(json.loads(t.to_json()))
        if t.__dict__ != tt.__dict__:
            print('MISMATCH:')
            print('   ', tt.to_json())
        t.plot(title=fn)


if __name__ == '__main__':
    test(sys.argv[1:])
