from numbers import Number
from typing import List, Tuple
import matplotlib.pyplot as plt
from math import *

from anim.geom import Point, Vector, Line
from anim.transform import Transform
from rack import Rack


def circle(r, c=(0, 0)):
    cx, cy = c
    pt = lambda t: (r * sin(t) + cx, r * cos(t) + cy)
    steps = 360
    return [pt(step / steps * tau) for step in range(steps + 1)]


def plot(xy, color='black'):
    plt.plot(*zip(*xy), color)


class Involute(object):
    """
        Involute curve.
        See https://en.wikipedia.org/wiki/Involute#Involutes_of_a_circle
    """

    def __init__(self, radius, max_radius, min_radius=0.0):
        self.radius = radius
        self.max_radius = max_radius
        if min_radius <= radius:
            self.clipped = False
            self.min_radius = radius
        else:
            self.clipped = True
            self.min_radius = min_radius
        self.start_angle = self.calc_angle(self.min_radius)
        self.end_angle = self.calc_angle(self.max_radius)
        print('Inv: r=%.6f mnr=%.6f mxr=%.6f sa=%.6f ea=%.6f' %
              (self.radius, self.min_radius, self.max_radius, self.start_angle, self.end_angle))

    def calc_angle(self, distance) -> float:
        """Calculate angle (radians) for corresponding distance (> radius)"""
        assert (distance >= self.radius)
        return sqrt(distance * distance / (self.radius * self.radius) - 1)

    def calc_point(self, angle, offset=0.0):
        """Calculate the x,y for a given angle and offset angle"""
        x = self.radius * (cos(angle) + (angle - offset) * sin(angle))
        y = self.radius * (sin(angle) - (angle - offset) * cos(angle))
        return x, y

    def path(self, steps=10, offset=0.0, up=1, center=(0.0, 0.0)) -> List[Tuple[Number, Number]]:
        """Generate path for involute"""
        factor = up * (self.end_angle - self.start_angle) / steps
        sa = up * self.start_angle
        zp = (self.calc_point(factor * step + offset + sa, offset=offset) for step in range(0, steps + 1))
        cx, cy = center
        return [(x + cx, y + cy) for x, y in zp]


class Gear(object):
    def __init__(self, teeth=30, center=(0.0, 0.0), rot=0.0, module=1.0,
                 pressure_angle=20.0, pressure_line=True):
        """
            Plot a gear
            :param teeth:	Number of teeth in gear
            :param center:  Center of gear
            :param rot: 	Rotation in #teeth
            :param module:	Module of gear
            :param pressure_angle: Pressure angle
            :param pressure_line: True to plot pressure lines
        """
        self.teeth = teeth
        self.module = module
        self.center = center
        self.pitch = self.module * pi
        self.rot = rot * self.pitch  # Now rotation is in pitch distance
        self.relief_factor = 1.25
        self.pressure_angle = radians(pressure_angle)
        self.pressure_line = pressure_line
        self.pitch_radius = self.module * self.teeth / 2
        self.base_radius = self.pitch_radius * cos(self.pressure_angle)
        self.tip_radius = self.pitch_radius + self.module   # add addendum
        print('pr=%8.6f br=%8.6f cpa=%9.7f' % (self.pitch_radius, self.base_radius, cos(self.pressure_angle)))

    def gen_poly(self) -> List[Tuple[Number, Number]]:
        addendum = self.module
        dedendum = self.module * 1.25
        tooth = self.pitch / 2
        addendum_offset = addendum * tan(self.pressure_angle)
        dedendum_offset = dedendum * tan(self.pressure_angle)
        print(self.pitch, tooth, addendum_offset)
        print('pitch=', self.pitch, ' cp=', tau / self.teeth)
        points = []
        br = self.base_radius
        pr = self.pitch_radius
        dr = self.pitch_radius - dedendum
        cx, cy = self.center
        inv = Involute(self.base_radius, self.pitch_radius + addendum, dr)
        # Calc pitch point where involute intersects pitch circle and offset
        pp_inv_angle = inv.calc_angle(self.pitch_radius)
        ppx, ppy = inv.calc_point(pp_inv_angle)
        pp_off_angle = atan2(ppy, ppx)
        # Multiply pp_off_angle by pr to move from angular to pitch space
        tooth_offset = tooth / 2 - pp_off_angle * pr

        for n in range(self.teeth):
            mid = n * self.pitch + self.rot

            start_angle = (mid - tooth_offset) / pr
            pts = inv.path(offset=start_angle, center=self.center, up=-1)
            points.extend(reversed(pts))
            if not inv.clipped:
                points.append((dr * cos(start_angle) + cx, dr * sin(start_angle) + cy))

            start_angle = (mid + tooth_offset) / pr
            if not inv.clipped:
                points.append((dr * cos(start_angle) + cx, dr * sin(start_angle) + cy))
            pts = inv.path(offset=start_angle, center=self.center, up=1)
            points.extend(pts)
        points.append(points[0])

        return points

    def gen_by_rack(self):
        """Generate the gear shape by moving a rack past the gear"""
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle),
                    relief_factor=self.relief_factor, tall_tooth=True)
        gear_points = []
        steps = 50
        z_teeth = 3
        rack_x = self.pitch_radius + self.center[0]
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        only_one_edge = False
        if only_one_edge:
            tooth_pts = tooth_pts[:2]
        for step in range(-steps, steps+1):
            tooth_pos = z_teeth * step / steps
            rack_y = tooth_pos * self.pitch + self.center[1]
            rack_pos = Vector(rack_x, rack_y)
            gear_rotation = tooth_pos / self.teeth * 360
            # Now, rotate tooth points back into default gear rotation
            t = Transform().rotate_about(gear_rotation, Point(*self.center))
            for pt in tooth_pts:
                gear_points.append(t.transform_pt(pt+rack_pos))
        return gear_points

    def gen_cuts_by_rack(self) -> Tuple[List[Line], List[Line]]:
        """
            Generate a set of cuts by moving a rack past the gear.
            :return: two lists of segments for cuts
        """
        # TODO-if we require center to be 0,0 for generating cuts, then low cuts is reflection of high cuts
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle),
                    relief_factor=self.relief_factor, tall_tooth=True)
        high_cuts = []
        low_cuts = []
        steps = 25
        z_teeth = 3
        center = Point(*self.center)
        rack_x = self.pitch_radius + center.x
        tooth_pts = [rack.tooth_tip_high, rack.tooth_base_high, rack.tooth_tip_low, rack.tooth_base_low]
        for step in range(-steps, steps+1):
            tooth_pos = z_teeth * step / steps
            rack_y = tooth_pos * self.pitch + self.center[1]
            rack_pos = Vector(rack_x, rack_y)
            gear_rotation = tooth_pos / self.teeth * 360
            # Now, rotate tooth points back into default gear rotation and generate segments
            t = Transform().rotate_about(gear_rotation, center)
            tth, tbh, ttl, tbl = (t.transform_pt(pt+rack_pos, True) for pt in tooth_pts)
            # Filter out unnecessary cuts
            # TODO-this optimization appears to work as long as #teeth >= 7
            #   It could probably be base_radius when #teeth > 20
            if (tth - center).length() <= (self.tip_radius if step < 0 else self.pitch_radius):
                high_cuts.append(Line.from_pts(tth, tbh))
            if (ttl - center).length() <= (self.tip_radius if step > 0 else self.pitch_radius):
                low_cuts.append(Line.from_pts(ttl, tbl))
        # Cut from flattest to steepest
        low_cuts = list(reversed(low_cuts))
        return high_cuts, low_cuts

    def cuts_for_mill(self, tool_angle) -> List[Tuple[float, float, float]]:
        """
            Generate raw coordinates for milling.

            :return: [(rotation-in-degrees, depth-into-gear, height),] aka a, y, z
        """
        assert(self.center == (0, 0))
        high_cuts, low_cuts = self.gen_cuts_by_rack()
        cut_params = []
        for cut in high_cuts:
            cut_angle = cut.direction.angle()
            rotation = tool_angle-cut_angle
            print('ca=%9.6f rot=%9.6f' % (cut_angle, rotation))
            t = Transform().rotate(rotation)
            y, z = t.transform_pt(cut.origin)
            cut_params.append((rotation, y, z))
        return cut_params

    def gen_tooth(self):
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle), relief_factor=self.relief_factor)
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        offset = Vector(self.pitch_radius, 0) + Vector(*self.center)
        return [(p+offset).xy() for p in tooth_pts]

    def plot(self, color='red'):
        addendum = self.module
        dedendum = self.module * 1.25
        pitch_radius = self.pitch_radius

        if self.pressure_line:
            dx = self.module*5*sin(self.pressure_angle)
            dy = self.module*5*cos(self.pressure_angle)
            cx = pitch_radius + self.center[0]
            cy = 0 + self.center[1]
            plot([(cx+dx, cy+dy), (cx-dx, cy-dy)], color='#FFA0FF')
            plot([(cx-dx, cy+dy), (cx+dx, cy-dy)], color='#FFA0FF')

        plot(circle(pitch_radius, c=self.center), color='green')
        plot(circle(pitch_radius + addendum, c=self.center), color='yellow')
        plot(circle(pitch_radius - dedendum, c=self.center), color='blue')
        plot(circle(pitch_radius - addendum, c=self.center), color='cyan')
        plot(circle(self.base_radius, c=self.center), color='orange')

        # plot(circle(2 * self.module, c=self.center), color='red')
        # plot(circle(self.module, c=self.center), color='blue')

        # plot(self.gen_by_rack(), color='#808080')
        plot_cuts_in_gear_space = True
        if plot_cuts_in_gear_space:
            origins = []
            high_cuts, low_cuts = self.gen_cuts_by_rack()
            col = '#808080'
            for idx, cut in enumerate(high_cuts):
                val = int(idx/len(high_cuts)*128+128)
                col = '#2020FF' if idx % 5 == 0 else '#%02x8080' % val
                if idx == len(high_cuts)-1:
                    col = 'orange'
                plot([cut.p1.xy(), cut.p2.xy()], col)
                origins.append(cut.origin)
            plot(origins, col)
            origins = []
            for idx, cut in enumerate(low_cuts):
                val = int(idx/len(low_cuts)*128+128)
                col = '#2020FF' if idx % 5 == 0 else '#80%02x80' % val
                if idx == len(low_cuts)-1:
                    col = 'orange'
                # plot([cut.p1.xy(), cut.p2.xy()], color)
                origins.append(cut.origin)
            plot(origins, col)
            plot(self.gen_tooth(), color='green')

        plot_cuts_in_mill_space = not True
        if plot_cuts_in_mill_space:
            tool_angle = 45.0
            cuts = self.cuts_for_mill(tool_angle/2)
            pts = []
            cut_vec_x = 2*cos(radians(tool_angle/2))
            cut_vec_y = 2*sin(radians(tool_angle/2))
            for rotation, y, z in cuts:
                rot = radians(rotation)
                plot([(0, 0), (pitch_radius*cos(rot)/2, pitch_radius*sin(rot)/2)], '#8080F0')
                plot([(y, z), (y+cut_vec_x, z+cut_vec_y)], '#808080')
                # pts.append((y, z))
            plot(pts, 'green')

        plot(self.gen_poly(), color=color)


def do_gears(rot=0., zoom_radius=0.):
    # print(circle(2))
    # plot(circle(1, (1, -.5)), color='blue')
    # rot = 0.25
    # rot = 0.0
    t1 = 17
    Gear(t1, rot=rot, module=1).plot()
    print()
    # gear(30)
    t2 = 5
    # Gear(t2, center=((t1 + t2) / 2, 0), rot=-rot, pressure_line=False).plot()
    plt.axis('equal')
    plt.grid()
    # Set zoom_radius to zoom in around where gears meet
    if zoom_radius:
        ax = plt.gca()
        ax.set_xlim(t1 / 2 - zoom_radius, t1 / 2 + zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
    plt.show()


def main():
    # do_gears(0); return

    do_gears(0, zoom_radius=8)
    do_gears(0, zoom_radius=4)
    do_gears(0, zoom_radius=2)
    do_gears(0, zoom_radius=1)
    return

    # for rot in (n/20 for n in range(21)):
    for rot in [0, 0.125, 0.25, 0.375, 0.5]:
        do_gears(rot=rot, zoom_radius=5)

    show_unzoomed = False
    if show_unzoomed:
        for rot in [0, 0.125, 0.25, 0.375, 0.5]:
            do_gears(rot=rot)


if __name__ == '__main__':
    main()