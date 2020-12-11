from math import pi, cos, sin, atan2, sqrt, tau
from typing import Tuple, cast, Union

from matplotlib import pyplot as plt

from anim.geom import Point, BasePoint
from gear_base import PointList, t_range, path_rotate, GearInstance, plot, circle


class CycloidalPair:
    """Cycloidal pair of gears"""
    def __init__(self,
                 wheel_teeth=30, pinion_teeth=8, module=1.0,        # Definitional attributes
                 generating_radius=0.0,                             # Defaults to pinion_radius/2
                 ):
        """
            Plot a cycloidal gear
            :param wheel_teeth:	Number of teeth in gear
            :param module:	Module of gear
        """
        self.module = module
        self.wheel_teeth = wheel_teeth
        self.pinion_teeth = pinion_teeth

        self.circular_pitch = self.module * pi
        self.pitch_diameter = self.module * self.wheel_teeth
        self.pitch_radius = self.pitch_diameter / 2
        self.wheel_pitch_radius = self.module * self.wheel_teeth / 2
        self.pinion_pitch_radius = self.module * self.pinion_teeth / 2
        self.wheel_tooth_degrees = 360 / self.wheel_teeth
        self.pinion_tooth_degrees = 360 / self.pinion_teeth

        self.generating_radius = generating_radius or self.pinion_pitch_radius / 2

        self.pitch = self.module * pi
        a_f_t, theta = self.calc_addendum_factor()
        self.addendum_factor_theoretical = a_f_t
        self.addendum_factor = 0.95 * self.addendum_factor_theoretical
        self.wheel_tooth_theta = theta
        self.wheel_addendum = self.addendum_factor * self.module
        fa, fr = self.pinion_factors()
        self.pinion_addendum_factor = fa
        self.pinion_addendum_radius = fr
        self.wheel_dedendum = self.pinion_addendum_factor * self.module
        self.pinion_dedendum = self.addendum_factor_theoretical * 1.05 * self.module
        self.wheel_base_radius = self.wheel_pitch_radius - self.wheel_dedendum * 1.05
        self.pinion_base_radius = self.pinion_pitch_radius - self.wheel_addendum * 1.4
        self.wheel_tip_radius = self.pitch_radius + self.wheel_addendum
        self.pinion_tip_radius = self.pinion_pitch_radius + self.pinion_addendum_factor * self.module
        # print('pr=%8.6f af=%8.6f' % (self.pitch_radius, self.addendum_factor))

    def __str__(self):
        return 'CycloidalPair: wheel=%d pinion=%d module=%g' % (self.wheel_teeth, self.pinion_teeth, self.module)

    def calc_cycloid(self, theta) -> Point:
        # https://en.wikipedia.org/wiki/Epicycloid
        rr = self.generating_radius + self.wheel_pitch_radius
        return Point(
            rr * cos(theta) - self.generating_radius * cos(rr / self.generating_radius * theta),
            rr * sin(theta) - self.generating_radius * sin(rr / self.generating_radius * theta)
        )

    def cycloid_path(self, theta_min=0.0, theta_max=pi/2, steps=5) -> PointList:
        return [self.calc_cycloid(theta) for theta in t_range(steps, theta_min, theta_max)]

    def calc_addendum_factor(self) -> Tuple[float, float]:
        """Calculate addendum height and corresponding theta"""
        # Equations from https://www.csparks.com/watchmaking/CycloidalGears/index.jxl

        pr_gr = self.wheel_pitch_radius / self.generating_radius

        def theta_func(beta):
            return pi / self.pinion_teeth + pr_gr * beta

        def beta_func(theta):
            return atan2(sin(theta), (1 + pr_gr - cos(theta)))

        theta_guess = 1.0
        beta_guess = 0.0
        error = 1.0
        max_error = 1e-10
        while error > max_error:
            beta_guess = beta_func(theta_guess)
            theta_new = theta_func(beta_guess)
            error = abs(theta_new - theta_guess)
            theta_guess = theta_new
        k = 1 + pr_gr
        af = self.pinion_teeth/4 * (1 - k + sqrt(1 + k*k - 2*k*cos(theta_guess)))
        debug_stuff = False
        if debug_stuff:
            print('calc_af: af=%.4f beta=%.4f theta=%.4f' % (af, beta_guess, theta_guess))
            ht = tau / self.wheel_teeth / 4
            pt = self.calc_cycloid(ht)
            print('   : ht=%.4f cc=%s ccl=%.4f' % (ht, pt.round(4), (pt-Point(0, 0)).length()))
            print('   : ')
        return af, theta_guess / pr_gr

    def gen_wheel_tooth(self) -> PointList:
        half_tooth = self.wheel_tooth_degrees / 4
        cycloid_path_down = list(reversed(self.cycloid_path(theta_max=self.wheel_tooth_theta, steps=5)))
        cycloid_path_up = [Point(p.x, -p.y) for p in reversed(cycloid_path_down)]

        tooth_path = []
        tip_high = cast(PointList, path_rotate(cycloid_path_up, -half_tooth, True))
        tip_low = cast(PointList, path_rotate(cycloid_path_down, half_tooth, True))
        origin = Point(0, 0)

        def scale_pt(pt: BasePoint):
            return (pt-origin).unit()*self.wheel_base_radius + origin

        tooth_path.append(scale_pt(tip_high[0]))
        tooth_path.extend(tip_high)
        tooth_path.extend(tip_low[1:])      # First point is duplicate of tip_high last point
        tooth_path.append(scale_pt(tip_low[-1]))
        return tooth_path

    def gen_poly(self, rotation=0.0) -> PointList:
        rotation *= self.wheel_tooth_degrees
        tooth_path = self.gen_wheel_tooth()

        points = []
        # for n in [-1, 0, 1]:
        for n in range(self.wheel_teeth):
            mid = n * self.wheel_tooth_degrees + rotation
            points.extend(path_rotate(tooth_path, mid, True))
        # points.append(points[0])

        return points

    def pinion_factors(self) -> Tuple[float, float]:
        """
            Lookup the pinion addendum and radius factors based in number of teeth.
            :returns: f_addendum, f_radius

            Data from: https://www.csparks.com/watchmaking/CycloidalGears/index.jxl
        """
        if self.pinion_teeth < 8:
            return 0.855, 1.050
        elif self.pinion_teeth < 10:
            return 0.670, 0.700
        else:
            return 0.625, 0.625

    def gen_pinion_tooth(self) -> PointList:
        half_tooth = tau / self.pinion_teeth / 4
        if self.pinion_teeth < 10:
            half_tooth *= 1.05 / (pi/2)
            if self.pinion_teeth < 8:
                a, b, c, d = [fa*0.855/1.05 for fa in [0.25, 0.55, 0.8, 1.05]]
            else:
                a, b, c, d = [fa*0.670/1.05 for fa in [0.25, 0.55, 0.8, 1.05]]
        else:
            half_tooth *= 1.25 / (pi/2)
            a, b, c, d = [fa*0.625/0.55 for fa in [0.2, 0.35, 0.5, 0.55]]
        tooth_path_polar = [
            # radius          angle
            (self.pinion_base_radius, 2.5*half_tooth),
            (self.pinion_base_radius, half_tooth),
            (self.pinion_pitch_radius, half_tooth),
            (self.pinion_pitch_radius+a*self.module, half_tooth*0.9),
            (self.pinion_pitch_radius+b*self.module, half_tooth*0.7),
            (self.pinion_pitch_radius+c*self.module, half_tooth*0.4),
            (self.pinion_pitch_radius+d*self.module, 0),
            (self.pinion_pitch_radius+c*self.module, -half_tooth*0.4),
            (self.pinion_pitch_radius+b*self.module, -half_tooth*0.7),
            (self.pinion_pitch_radius+a*self.module, -half_tooth*0.9),
            (self.pinion_pitch_radius, -half_tooth),
            (self.pinion_base_radius, -half_tooth),
            (self.pinion_base_radius, -2.5*half_tooth),
        ]
        # Hack to chop off the tip, just for demo
        # tooth_path_polar = tooth_path_polar[:3] + tooth_path_polar[-3:]
        tooth_path = [Point(r*cos(t), r*sin(t)) for r, t in tooth_path_polar]
        return tooth_path

    def gen_pinion_poly(self, rotation=0.0) -> PointList:
        if self.pinion_teeth % 2 == 0:
            rotation += 0.5
        rotation *= self.pinion_tooth_degrees
        tooth_path = self.gen_pinion_tooth()

        points = []
        # for n in [-1, 0, 1]:
        for n in range(self.pinion_teeth):
            mid = n * self.pinion_tooth_degrees + rotation
            rotated = path_rotate(tooth_path, mid, True)
            points.extend(rotated)

        # points.append(points[0])

        return points

    def wheel(self):
        """Return a gear instance that represents the wheel of the pair"""
        return GearInstance(self.module, self.wheel_teeth, 'Cycloidal', 'wheel', self.gen_poly(), Point(0, 0),
                            tip_radius=self.wheel_tip_radius, base_radius=self.wheel_base_radius)

    def pinion(self):
        """Return a gear instance that represents the pinion of the pair"""
        return GearInstance(self.module, self.pinion_teeth, 'Cycloidal', 'pinion', self.gen_pinion_poly(),
                            Point(self.wheel_pitch_radius+self.pinion_pitch_radius, 0),
                            tip_radius=self.pinion_tip_radius, base_radius=self.pinion_base_radius)

    def plot(self, color='blue', rotation=0.0, center=Point(0, 0), pinion: Union[str, bool] = True):
        addendum = self.module * self.addendum_factor_theoretical
        dedendum = self.pitch / 2
        pitch_radius = self.pitch_radius

        if pinion != 'only':
            self.wheel().plot(color, rotation=rotation)

        if pinion:
            p = self.pinion()
            p.plot(color, rotation=rotation)

        plt.title('%s' % self)
