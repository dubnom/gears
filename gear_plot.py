from numbers import Number
from typing import List, Tuple
import matplotlib.pyplot as plt
from math import *


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
        self.pressure_angle = radians(pressure_angle)
        self.pressure_line = pressure_line
        self.pitch_radius = self.module * self.teeth / 2
        self.base_radius = self.pitch_radius * cos(self.pressure_angle)
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

    def plot(self, color='red'):
        addendum = self.module
        dedendum = self.module * 1.25
        pitch_radius = self.module * self.teeth / 2

        if self.pressure_line:
            dx = self.module*5*sin(self.pressure_angle)
            dy = self.module*5*cos(self.pressure_angle)
            cx = pitch_radius + self.center[0]
            cy = 0 + self.center[1]
            plot([(cx+dx, cy+dy), (cx-dx, cy-dy)], color='purple')
            plot([(cx-dx, cy+dy), (cx+dx, cy-dy)], color='purple')

        plot(circle(pitch_radius, c=self.center), color='green')
        plot(circle(pitch_radius + addendum, c=self.center), color='yellow')
        plot(circle(pitch_radius - dedendum, c=self.center), color='blue')
        plot(circle(pitch_radius - addendum, c=self.center), color='cyan')
        plot(circle(self.base_radius, c=self.center), color='orange')

        # plot(circle(2 * self.module, c=self.center), color='red')
        # plot(circle(self.module, c=self.center), color='blue')

        plot(self.gen_poly(), color=color)


def do_gears(rot=0., zoom_radius=0.):
    # print(circle(2))
    # plot(circle(1, (1, -.5)), color='blue')
    # rot = 0.25
    # rot = 0.0
    t1 = 19
    Gear(t1, rot=rot, module=1).plot()
    print()
    # gear(30)
    t2 = 5
    Gear(t2, center=((t1 + t2) / 2, 0), rot=-rot, pressure_line=False).plot()
    plt.axis('equal')
    plt.grid()
    # Set zoom_radius to zoom in around where gears meet
    if zoom_radius:
        ax = plt.gca()
        ax.set_xlim(t1 / 2 - zoom_radius, t1 / 2 + zoom_radius)
        ax.set_ylim(-zoom_radius, zoom_radius)
    plt.show()


def main():
    # for rot in (n/20 for n in range(21)):
    for rot in [0, 0.125, 0.25, 0.375, 0.5]:
        do_gears(rot=rot, zoom_radius=5)

    show_unzoomed = False
    if show_unzoomed:
        for rot in [0, 0.125, 0.25, 0.375, 0.5]:
            do_gears(rot=rot)


if __name__ == '__main__':
    main()
