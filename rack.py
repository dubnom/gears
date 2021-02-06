"""
    Rack creation functions
"""

from math import pi, tan, radians
from typing import Tuple, List
from x7.geom.geom import Point, Line, Vector, PointOrXYList
from x7.geom.utils import path_to_xy, path_rotate_ccw, path_translate


class Rack(object):
    """
        Parameters and helpers for a rack.

        Rack modeling space:
            * (0,0) is on the pitch line centered on a tooth
            * rack is vertical (Y) with tooth pointing to the left (negative X)

    """

    def __init__(self, module=1, pressure_angle=20.0, relief_factor=1.25,
                 center=Point(0, 0), angle=0.0, pitch_radius=0.0,
                 tall_tooth=False):
        """
            :param module: gear module
            :param pressure_angle: angle of tooth
            :param relief_factor: extra depth for dedendum
            :param tall_tooth: True to make addendum equal dedendum for gear cutting simulations
        """
        self.module = module
        self.pressure_angle = pressure_angle
        self.relief_factor = relief_factor
        self.center = center
        self.angle = angle
        self.pitch_radius = pitch_radius

        self.circular_pitch = self.module * pi
        self.half_tooth = self.circular_pitch / 4

        dedendum = self.module * self.relief_factor
        addendum = self.module
        tan_pa = tan(radians(self.pressure_angle))
        if tall_tooth:
            addendum = dedendum
        tip_offset = self.half_tooth - addendum*tan_pa
        base_offset = self.half_tooth + dedendum*tan_pa
        self.tooth_tip_high = Point(-addendum, tip_offset)
        self.tooth_tip_low = Point(-addendum, -tip_offset)
        self.tooth_base_high = Point(dedendum, base_offset)
        self.tooth_base_low = Point(dedendum, -base_offset)
        self.tooth_edge_high = Line.from_pts(self.tooth_tip_high, self.tooth_base_high)
        self.tooth_edge_low = Line.from_pts(self.tooth_tip_low, self.tooth_base_low)

    def points(self, teeth_in_gear=32, teeth_in_rack=10) -> List[Tuple[float, float]]:
        # TODO-retire this API in favor of .path()
        # def make_rack(module, num_teeth, h_total, half_tooth, pressure_angle):
        """
            :param teeth_in_gear: number of teeth in gear (used to locate rack horizontally)
            :param teeth_in_rack: duh, ignored
            :return: list of x,y pairs
        """

        pitch_radius = self.module * teeth_in_gear / 2
        w_tooth = self.half_tooth * 2
        rack_pts = []
        tooth_pts = [self.tooth_base_high, self.tooth_tip_high, self.tooth_tip_low, self.tooth_base_low]
        steps = (teeth_in_rack+1)//2
        for step in range(-steps, steps+1):
            offset = Vector(pitch_radius, -step*self.circular_pitch)
            for tp in tooth_pts:
                rack_pts.append((tp+offset).xy())

        # Add a back to the rack to make a closed polygon
        top = rack_pts[-1][1]
        bot = rack_pts[0][1]
        back = self.module * 4 + pitch_radius
        rack_pts.extend([(back, top), (back, bot)])

        return rack_pts

    def path(self, teeth=1, rot=0.5, closed=False, depth=8, as_pt=True) -> PointOrXYList:
        """
            Generate a rack
            :param teeth:   Number of teeth (must be odd)
            :param rot:     Rotation of gear (shifts rack accordingly)
            :param closed:  True to 'close' the rack around the back
            :param depth:   Depth of closure in module-units
            :param as_pt:   True to return points
            :return:
        """
        tooth_pts = [self.tooth_base_high, self.tooth_tip_high, self.tooth_tip_low, self.tooth_base_low]
        y_shift = Vector(0, -self.circular_pitch)
        offset = Vector(self.pitch_radius, 0) + y_shift * (rot % 1)
        teeth //= 2
        path = [(p+offset+y_shift*tooth) for tooth in range(-teeth, teeth+1) for p in tooth_pts]
        if closed:
            # Add a back to the rack to make a closed polygon
            top = path[-1].y
            bot = path[0].y
            back = self.module * depth + self.pitch_radius
            path.extend([Point(back, top), Point(back, bot)])
        if self.angle:
            path = path_rotate_ccw(path, self.angle, as_pt=True)
        path = path_translate(path, self.center, as_pt=True)
        return path if as_pt else path_to_xy(path)

        # One tooth
        #      ____          +h_a    # addendum
        #     /    \
        #    /      \          0     # pitch line / pitch radius
        #   /        \
        #  /          \____  -h_d    # dedendum
        #  x0  x1  x2  x3  x4
        #  x0                 is 0
        #  <>   is tan(pressure_angle) * h_d => w_d
        #    <> is tan(pressure_angle) * h_a => w_a
        #  <--> is tan(pressure_angle) * h_t == w_d + w_a => w_ad   |x1-x0| and |x3-x2|
        #    <------> is w_tooth (along pitch radius)
        #        <------> is also w_tooth (middle of top to middle of bottom)
        #      <--> is w_tooth - 2 * w_a) => w_top                  |x2-x1|
        #              <--> is w_tooth - 2 * w_d) => w_bot          |x4-x3|
