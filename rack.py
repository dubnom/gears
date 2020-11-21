"""
    Rack creation functions
"""

from math import pi, tan, degrees, radians
from typing import Tuple, List


class Rack(object):
    """Parameters and helpers for a rack"""

    def __init__(self, module=1, pressure_angle=20.0, relief_factor=1.25):
        """
            :param module: gear module
            :param pressure_angle: angle of tooth
            :param relief_factor: extra depth for dedendum
        """
        self.module = module
        self.pressure_angle = pressure_angle
        self.relief_factor = relief_factor
        self.circular_pitch = self.module * pi
        self.half_tooth = self.circular_pitch / 4

    def cut_points(self) -> List[Tuple[float, float]]:
        """Location in tooth space of cut points [high, low]"""
        dedendum = self.module * self.relief_factor
        pressure_offset = dedendum * tan(radians(self.pressure_angle))
        cut_high = self.half_tooth - pressure_offset, dedendum
        cut_low = self.half_tooth + pressure_offset, -dedendum
        return [cut_high, cut_low]

    def points(self, teeth_in_gear=32, teeth_in_rack=10):
        # def make_rack(module, num_teeth, h_total, half_tooth, pressure_angle):
        """
            :param teeth_in_gear: number of teeth in gear (used to locate rack horizontally)
            :param teeth_in_rack: duh
            :return: list of x,y pairs
        """

        pitch_radius = self.module * teeth_in_gear / 2
        w_tooth = self.half_tooth * 2
        rack_pts = []
        h_a = self.module
        h_d = self.module * self.relief_factor
        h_t = h_a + h_d

        # One tooth
        #      ____          +h_a    # addendum
        #     /    \
        #    /      \          0
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
        # TODO-Add divots at the bottom of the tooth are added to aid visual tracking
        tan_p = tan(radians(self.pressure_angle))
        w_d = tan_p * h_d
        w_a = tan_p * h_a
        w_ad = w_a + w_d
        w_top = w_tooth - 2 * w_a
        w_bot = w_tooth - 2 * w_d

        tooth_pts = [(0, -h_d), (w_ad, h_a), (w_ad + w_top, h_a), (w_ad + w_top + w_ad, -h_d),
                     (2 * w_tooth, -h_d)]
        for rack_tooth in range(-10, 5 + 1):
            offset = rack_tooth * 2 * w_tooth + w_ad + w_top / 2
            rack_pts.extend([(pitch_radius - ty, pitch_radius + tx + offset) for tx, ty in tooth_pts])

        # Add a back to the rack to make a closed polygon
        top = rack_pts[-1][1]
        bot = rack_pts[0][1]
        back = self.module * 4 + pitch_radius
        rack_pts.extend([(back, top), (back, bot)])

        return rack_pts


def make_rack(module, num_teeth, h_total, half_tooth, pressure_angle):
    """
    Create a list of x, y pairs representing a rack in the Y direction

    :param module: gear module
    :param num_teeth: number of teeth on gear
    :param h_total: height of addendum + dedendum
    :param half_tooth: width of half a tooth
    :param pressure_angle: angle of tooth
    """

    pitch_radius = module * num_teeth / 2
    w_tooth = half_tooth * 2
    rack_pts = []
    h_a = module
    h_d = module * 1.25
    h_t = h_a + h_d
    assert (h_t == h_total)

    # One tooth
    #      ____          +h_a    # addendum
    #     /    \
    #    /      \          0
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
    # TODO-Add divots at the bottom of the tooth are added to aid visual tracking
    tan_p = tan(radians(pressure_angle))
    w_d = tan_p * h_d
    w_a = tan_p * h_a
    w_ad = w_a + w_d
    w_top = w_tooth - 2 * w_a
    w_bot = w_tooth - 2 * w_d

    tooth_pts = [(0, -h_d), (w_ad, h_a), (w_ad + w_top, h_a), (w_ad + w_top + w_ad, -h_d),
                 (2 * w_tooth, -h_d)]
    for rack_tooth in range(-10, 5 + 1):
        offset = rack_tooth * 2 * w_tooth + w_ad + w_top / 2
        rack_pts.extend([(pitch_radius - ty, pitch_radius + tx + offset) for tx, ty in tooth_pts])

    # Add a back to the rack to make a closed polygon
    top = rack_pts[-1][1]
    bot = rack_pts[0][1]
    back = module * 4 + pitch_radius
    rack_pts.extend([(back, top), (back, bot)])

    return rack_pts
