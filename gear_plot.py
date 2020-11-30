import os
from numbers import Number
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
from math import *

from anim.geom import Point, Vector, Line, BasePoint
from anim.transform import Transform
from rack import Rack

# setenv SHOW_INTERACTIVE to 1 or true to display interactive plots
SHOW_INTERACTIVE = os.environ.get('SHOW_INTERACTIVE', 'false').lower() in {'1', 'true'}


def t_range(steps, t_low=0.0, t_high=1.0, closed=True):
    """Range from t_low to t_high in steps.  If closed, include t_high"""
    t_len = t_high - t_low
    return (step / steps * t_len + t_low for step in range(0, steps + int(closed)))


def path_rotate(path, angle, as_pt=False):
    """Rotate all points by angle (in degrees) about 0,0"""
    t = Transform().rotate(angle)
    return [t.transform_pt(pt, as_pt) for pt in path]


def path_translate(path, dx, dy, as_pt=False):
    """Rotate all points by angle (in degrees) about 0,0"""
    return [(x+dx, y+dy) for x, y in path]


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
    def __init__(self, teeth=30, center=(0.0, 0.0), rot=0.0,
                 module=1.0, relief_factor=1.25,
                 pressure_angle=20.0, pressure_line=True):
        """
            Plot an involute gear
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
        self.relief_factor = relief_factor
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
        # If we require center to be 0,0 for generating cuts, then low cuts is reflection of high cuts
        assert(self.center == (0, 0))
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle),
                    relief_factor=self.relief_factor, tall_tooth=True)
        high_cuts = []
        # TODO-this should be calculated based on when the rack intersects the tip circle
        #     -actually, it can be calculated based on the first and last intersection with the pressure line
        #     -for large gears, this will be a long track
        #     -consider having smaller steps near the center of the gear as these steps likely matter more
        z_teeth = 35
        steps = z_teeth * 4
        rack_x = self.pitch_radius
        overshoot = self.module * 0.25
        pressure_center = Point(self.pitch_radius, 0)
        pressure_vector_high = Vector(-sin(self.pressure_angle), cos(self.pressure_angle))*5000
        pressure_line_high = Line(pressure_center-pressure_vector_high, 2*pressure_vector_high)
        center = Point(0, 0)
        for step in range(-steps, steps+1):
            tooth_pos = z_teeth * step / steps
            rack_y = tooth_pos * self.pitch
            rack_pos = Vector(rack_x, rack_y)
            gear_rotation = tooth_pos / self.teeth * 360
            rack_tooth_edge_high = rack.tooth_edge_high + rack_pos
            # The cut never needs to cross the pressure line, since that
            # is where the rack contacts the gear.
            intersection = pressure_line_high.segment_intersection(rack_tooth_edge_high)
            tth_max = rack.tooth_tip_high+rack_pos
            tth = intersection[0] if intersection else tth_max
            if pressure_line_high.parallel(rack_tooth_edge_high):
                print('%4d: Parallel' % step)
            # where, t1, t2 = intersection
            # print('intersection: %.4f,%.4f %.4f %.4f' % (where.x, where.y, t1, t2))
            tbh = rack.tooth_base_high+rack_pos

            # Now, rotate tooth points back into default gear rotation and generate segments
            t = Transform().rotate_about(gear_rotation, center)
            tth, tbh, tth_max = (t.transform_pt(pt, True) for pt in [tth, tbh, tth_max])

            # Filter out unnecessary cuts
            # TODO-what about undercut?
            if intersection:
                # This looks like a good cut, so compute cut with small amount of overshoot
                extra = tth_max - tth
                if extra.length() < overshoot:
                    tth = tth_max
                else:
                    tth = tth + extra.unit() * overshoot
                high_cuts.append(Line.from_pts(tth, tbh))

        # Cut from flattest to steepest
        low_cuts = [Line(Point(hc.origin.x, -hc.origin.y), Vector(hc.direction.x, -hc.direction.y)) for hc in high_cuts]
        return high_cuts, low_cuts

    def cuts_for_mill(self, tool_angle, tool_tip_height=0.0) -> List[Tuple[float, float, float]]:
        """
            Generate raw coordinates for milling.

            :return: [(rotation-in-degrees, height, depth-into-gear),] aka a, y, z
        """
        # print('cuts_for_mill: ta=%.3f tth=%.3f' % (tool_angle, tool_tip_height))
        tool_angle /= 2
        # tool_tip_height = 0
        half_tool_tip = tool_tip_height/2
        assert(self.center == (0, 0))

        high_cuts, low_cuts = self.gen_cuts_by_rack()

        # Clear the center
        # Distance is always
        center_params = []
        root_top_cut = Line(high_cuts[-1].origin, Vector(1, 0))
        root_width = (high_cuts[-1].origin-low_cuts[-1].origin).length()
        root_cut_width = root_width-tool_tip_height
        # TODO-take into account tool_angle and pointy tools
        assert tool_angle == 0.0
        assert tool_tip_height != 0.0
        root_steps = int(root_width / tool_tip_height)
        root_depth = self.pitch_radius-self.module*self.relief_factor
        for step in range(0, root_steps+1):
            z = root_top_cut.origin.y - step/root_steps*root_cut_width
            # center_params.append((0, root_depth, z))
            center_params.append((0, root_depth, z-half_tool_tip))

        # Clear the tip
        tip_params = []
        tooth_angle = 360 / self.teeth
        t = Transform().rotate(tooth_angle)
        tip_cut = Line(low_cuts[-1].p2, t.transform_pt(high_cuts[-1].p2, True))
        tip_width = self.pitch / 4
        tip_cut_width = tip_width
        # TODO-take into account tool_angle and pointy tools
        assert tool_angle == 0.0
        assert tool_tip_height != 0.0
        tip_steps = int(tip_width / tool_tip_height / 2) + 1
        tip_height = self.pitch_radius+self.module
        tip_angle = tooth_angle / 2
        for step in range(-tip_steps, tip_steps+1):
            z = 0 - step/tip_steps*tip_cut_width/2
            tip_params.append((tip_angle, tip_height, z-half_tool_tip))


        high_params = []
        low_params = []
        for cut in high_cuts:
            cut_angle = cut.direction.angle()
            rotation = tool_angle-cut_angle
            # print('ca=%9.6f rot=%9.6f' % (cut_angle, rotation))
            t = Transform().rotate(-rotation)
            y, z = t.transform_pt(cut.origin)
            high_params.append((rotation, y, z-half_tool_tip))
        for cut in reversed(low_cuts):
            cut_angle = cut.direction.angle()
            rotation = -tool_angle-cut_angle
            # print('ca=%9.6f rot=%9.6f' % (cut_angle, rotation))
            t = Transform().rotate(-rotation)
            y, z = t.transform_pt(cut.origin)
            low_params.append((rotation, y, z+half_tool_tip))
        return high_params + center_params + list(reversed(low_params)) + tip_params

    def gen_tooth(self):
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle), relief_factor=self.relief_factor)
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        offset = Vector(self.pitch_radius, 0) + Vector(*self.center)
        return [(p+offset).xy() for p in tooth_pts]

    def plot(self, color='red', tool_angle=40.0, gear_space=None, mill_space=None):
        addendum = self.module
        dedendum = self.module * 1.25
        pitch_radius = self.pitch_radius
        if mill_space is None and gear_space is None:
            gear_space = True

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
        if gear_space:
            origins = []
            high_cuts, low_cuts = self.gen_cuts_by_rack()
            col = '#808080'
            for idx, cut in enumerate(high_cuts):
                val = int(idx/len(high_cuts)*128+128)
                col = '#2020FF' if idx % 5 == 0 else '#%02x8080' % val
                if idx == len(high_cuts)-1:
                    col = 'orange'
                plot([cut.p2.xy(), cut.p1.xy(), (cut.p1+cut.direction.normal().unit()*0.25).xy()], col)
                origins.append(cut.origin)
            # plot(origins, col)
            origins = []
            for idx, cut in enumerate(low_cuts):
                val = int(idx/len(low_cuts)*128+128)
                col = '#2020FF' if idx % 5 == 0 else '#80%02x80' % val
                if idx == len(low_cuts)-1:
                    col = 'orange'
                # plot([cut.p1.xy(), cut.p2.xy()], color)
                origins.append(cut.origin)
            plot(origins, col)
            plot(self.gen_tooth(), 'green')

        if mill_space:
            cuts = self.cuts_for_mill(tool_angle)
            pts = []
            cut_vec_z = 2*cos(radians(tool_angle/2))
            cut_vec_y = 2*sin(radians(tool_angle/2))
            for idx, (rotation, y, z) in enumerate(cuts):
                rot = radians(rotation)
                plot([
                    (0, 0),
                    (pitch_radius*cos(rot)*.8, pitch_radius*sin(rot)*.8),
                    (z, y)
                ], '#DDDDDD')
            for idx, (rotation, y, z) in enumerate(cuts):
                rot = radians(rotation)
                if idx % 5 == 0:
                    plot([
                        (0, 0),
                        (pitch_radius*cos(rot)*.8, pitch_radius*sin(rot)*.8),
                        (z, y)
                    ], '#8080F0')
                plot([(z, y), (z+cut_vec_z, y+cut_vec_y)], '#808080')
                # print('rot: %9.5f  z: %9.5f  y: %9.5f' % (rotation, z, y))
            plot(pts, 'green')

        plot(self.gen_poly(), color=color)
        # self.gen_gcode()

    def plot_show(self, zoom_radius=0):
        plt.axis('equal')
        plt.grid()
        # Set zoom_radius to zoom in around where gears meet
        if zoom_radius:
            ax = plt.gca()
            ax.set_xlim(self.teeth / 2 - zoom_radius, self.teeth / 2 + zoom_radius)
            ax.set_ylim(-zoom_radius, zoom_radius)
        plt.show()


class GearBase:
    def __init__(self, module, teeth):
        self.module = module
        self.teeth = teeth
        self.circular_pitch = self.module * pi
        self.pitch_diameter = self.module * self.teeth
        self.pitch_radius = self.pitch_diameter / 2

    def gen_poly(self):
        """Generate the polygon of the gear at (0,0) with 0 rotation"""
        ...

    def plot(self, zoom_radius=0.0):
        """Plot the gear poly and associated construction lines/circles"""
        ...

    def plot_show(self, zoom_radius=0.0):
        plt.axis('equal')
        plt.grid()
        # Set zoom_radius to zoom in around where gears meet
        if zoom_radius:
            ax = plt.gca()
            ax.set_xlim(self.teeth / 2 - zoom_radius, self.teeth / 2 + zoom_radius)
            ax.set_ylim(-zoom_radius, zoom_radius)
        plt.show()


class GearCycloidal(GearBase):
    def __init__(self,
                 module=1.0, wheel_teeth=30, pinion_teeth=8,        # Definitional attributes
                 generating_radius=0.0,                             # Defaults to pinion_radius/2
                 ):
        """
            Plot a cycloidal gear
            :param teeth:	Number of teeth in gear
            :param module:	Module of gear
            :param relief_factor: extra for dedendum
        """
        super().__init__(module, wheel_teeth)
        self.wheel_teeth = wheel_teeth
        self.pinion_teeth = pinion_teeth
        self.wheel_pitch_radius = self.module * self.wheel_teeth / 2
        self.pinion_pitch_radius = self.module * self.pinion_teeth / 2
        self.wheel_tooth_degrees = 360 / self.wheel_teeth
        self.pinion_tooth_degrees = 360 / self.pinion_teeth

        self.generating_radius = generating_radius or self.pinion_pitch_radius / 2

        self.pitch = self.module * pi
        aft, theta = self.calc_addendum_factor()
        self.addendum_factor_theoretical = aft
        self.addendum_factor = 0.95 * self.addendum_factor_theoretical
        self.wheel_tooth_theta = theta
        self.addendum = self.addendum_factor * self.module
        self.dedendum = self.addendum_factor_theoretical * 1.05 * self.module
        self.wheel_base_radius = self.wheel_pitch_radius - self.dedendum
        self.pinion_base_radius = self.pinion_pitch_radius - (self.addendum_factor + 0.4) * self.module
        self.tip_radius = self.pitch_radius + self.addendum
        # print('pr=%8.6f af=%8.6f' % (self.pitch_radius, self.addendum_factor))

    def calc_cycloid(self, theta) -> Point:
        # https://en.wikipedia.org/wiki/Epicycloid
        rr = self.generating_radius + self.wheel_pitch_radius
        return Point(
            rr * cos(theta) - self.generating_radius * cos(rr / self.generating_radius * theta),
            rr * sin(theta) - self.generating_radius * sin(rr / self.generating_radius * theta)
        )

    def cycloid_path(self, theta_min=0.0, theta_max=pi/2, steps=5) -> List[BasePoint]:
        vals = ['%.4f' % theta for theta in t_range(steps, theta_min, theta_max)]
        # print('cp: tm=%.4f %s' % (theta_max, ', '.join(vals)))
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

    def gen_poly(self, rotation=0.0) -> List[Tuple[Number, Number]]:
        rotation *= self.wheel_tooth_degrees
        half_tooth = self.wheel_tooth_degrees / 4
        cycloid_path_down = list(reversed(self.cycloid_path(theta_max=self.wheel_tooth_theta, steps=30)))
        cycloid_path_up = [Point(p.x, -p.y) for p in reversed(cycloid_path_down)]

        tooth_path = []
        tip_high = path_rotate(cycloid_path_up, - half_tooth, True)
        tip_low = path_rotate(cycloid_path_down, half_tooth, True)
        origin = Point(0, 0)

        def scale_pt(pt: Point):
            return (pt-origin).unit()*self.wheel_base_radius + origin

        tooth_path.append(scale_pt(tip_high[0]))
        tooth_path.extend(tip_high)
        tooth_path.extend(tip_low)
        tooth_path.append(scale_pt(tip_low[-1]))

        points = []
        # for n in [-1, 0, 1]:
        for n in range(self.teeth):
            mid = n * self.wheel_tooth_degrees + rotation
            points.extend(path_rotate(tooth_path, mid))
        points.append(points[0])

        return points

    def gen_pinion_poly(self, rotation=0.0, center=(0, 0)) -> List[Tuple[Number, Number]]:
        if self.pinion_teeth % 2 == 0:
            rotation += 0.5
        rotation *= self.pinion_tooth_degrees
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
        tooth_path_polar = tooth_path_polar
        tooth_path = [Point(r*cos(t), r*sin(t)) for r, t in tooth_path_polar]

        points = []
        # for n in [-1, 0, 1]:
        for n in range(self.teeth):
            mid = n * self.pinion_tooth_degrees + rotation
            points.extend(path_translate(path_rotate(tooth_path, mid), *center))

        # points.append(points[0])

        return points

    def plot(self, color='blue', rotation=0.0, center=Point(0, 0), pinion: Union[str, bool] = True):
        addendum = self.module * self.addendum_factor_theoretical
        dedendum = self.pitch / 2
        pitch_radius = self.pitch_radius

        if pinion != 'only':
            plot(circle(pitch_radius, c=center), color='green')
            plot(circle(pitch_radius + addendum, c=center), color='yellow')
            plot(circle(pitch_radius - dedendum, c=center), color='cyan')
            # plot(circle(pitch_radius - addendum, c=center), color='cyan')
            plot(self.gen_poly(rotation=rotation), color=color)

        if pinion:
            pinion_pitch = self.pinion_pitch_radius
            p_center = (pitch_radius+pinion_pitch, 0)
            plot(circle(pinion_pitch, c=p_center), color='green')
            plot(circle(pinion_pitch + addendum, c=p_center), color='yellow')
            plot(circle(self.pinion_base_radius, c=p_center), color='cyan')
            plot(self.gen_pinion_poly(rotation=-rotation, center=p_center), color=color)

        plt.title(r'$\sum_{idiot}^{\infty}sin(m^a_n)$')


def do_pinions(zoom_radius=0., cycloidal=True):
    wheel = None
    rot = 0.5
    color_index = 0
    colors = ['blue', 'red', 'green', 'orange']
    for pt in [6, 8, 10, 12]: #range(5, 11):
        # print(circle(2))
        # plot(circle(1, (1, -.5)), color='blue')
        # rot = 0.25
        # rot = 0.0
        t1 = 40
        t2 = pt
        # Gear(t1, rot=rot, module=1).plot(mill_space=True)
        if cycloidal:
            wheel = GearCycloidal(1, t1, t2)
            pinion = None
        else:
            wheel = Gear(t1, rot=rot, module=1)
            pinion = Gear(t2, center=((t1 + t2) / 2, 0), rot=-rot, pressure_line=False)

        wheel.plot(colors[color_index % len(colors)], rotation=rot, pinion='only')
        color_index += 1
        if pinion:
            pinion.plot('green')
    wheel.plot_show(zoom_radius)


def do_gears(rot=0., zoom_radius=0., cycloidal=True, pt=6):
    # print(circle(2))
    # plot(circle(1, (1, -.5)), color='blue')
    # rot = 0.25
    # rot = 0.0
    t1 = 40
    t2 = pt
    # Gear(t1, rot=rot, module=1).plot(mill_space=True)
    if cycloidal:
        wheel = GearCycloidal(1, t1, t2)
        pinion = None
    else:
        wheel = Gear(t1, rot=rot, module=1)
        pinion = Gear(t2, center=((t1 + t2) / 2, 0), rot=-rot, pressure_line=False)

    wheel.plot('blue', rotation=rot)
    if pinion:
        pinion.plot('green')
    wheel.plot_show(zoom_radius)


def main():
    do_pinions(zoom_radius=5); return
    for pt in [5, 6, 7, 8, 9, 10, 11]:
        do_gears(zoom_radius=5, pt=pt)
    return

    for rot in [0, 0.125, 0.25, 0.375, 0.5]:
        do_gears(rot=rot, zoom_radius=5)
    return

    do_gears(); return

    radii = [8, 4, 2, 1]
    radii = [8, 4]
    radii = [2]
    for radius in radii:
        do_gears(rot=0, zoom_radius=radius)
    return

    # for rot in (n/20 for n in range(21)):
    show_unzoomed = False
    if show_unzoomed:
        for rot in [0, 0.125, 0.25, 0.375, 0.5]:
            do_gears(rot=rot)


if __name__ == '__main__':
    main()
