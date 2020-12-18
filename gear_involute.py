from math import sqrt, cos, sin, pi, radians, tan, atan2, degrees
from numbers import Number
from typing import List, Tuple

from matplotlib import pyplot as plt

from anim.geom import Point, Vector, Line
from anim.transform import Transform
from gear_base import plot, GearInstance
from anim.utils import PointList, circle
from rack import Rack


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
        debug_print = False
        if debug_print:
            print('Inv: r=%.6f mnr=%.6f mxr=%.6f sa=%.6f ea=%.6f' %
                  (self.radius, self.min_radius, self.max_radius, self.start_angle, self.end_angle))

    def calc_angle(self, distance) -> float:
        """Calculate angle (radians) for corresponding distance (> radius)"""
        assert distance >= self.radius
        return sqrt(distance * distance / (self.radius * self.radius) - 1)

    def calc_point(self, angle, offset=0.0, offset_r=0.0, offset_n=0.0):
        """
            Calculate the x,y for a given angle and offset angle
            :param angle:       Angle in radians
            :param offset:      Angular offset in radians
            :param offset_r:    Offset along radial direction for final point
            :param offset_n:    Offset normal (CCW) to radial direction
            :return:
        """
        """"""
        # x = r * cos(a) + r*(a-oa) * sin(a)
        # x = (r-or) * cos(a) + r*(a-oa-on) * sin(a)
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


class GearInvolute(object):
    def __init__(self, teeth=30, center=Point(0, 0), rot=0.0,
                 module=1.0, relief_factor=1.25,
                 steps=4,
                 pressure_angle=20.0, pressure_line=True):
        """
            Plot an involute gear
            :param teeth:	Number of teeth in gear
            :param center:  Center of gear
            :param rot: 	Rotation in #teeth
            :param module:	Module of gear
            :param relief_factor: Relief factor
            :param steps:   Number of steps in the involute side
            :param pressure_angle: Pressure angle
            :param pressure_line: True to plot pressure lines
        """
        self.teeth = teeth
        self.module = module
        self.center = center
        self.pitch = self.module * pi
        self.rot = rot * self.pitch  # Now rotation is in pitch distance
        self.relief_factor = relief_factor
        self.steps = steps
        self.pressure_angle = radians(pressure_angle)
        self.pressure_line = pressure_line
        self.pitch_radius = self.module * self.teeth / 2
        self.base_radius = self.pitch_radius * cos(self.pressure_angle)
        self.tip_radius = self.pitch_radius + self.module   # add addendum
        self.dedendum_radius = self.pitch_radius - self.module * self.relief_factor
        # print('pr=%8.6f br=%8.6f cpa=%9.7f' % (self.pitch_radius, self.base_radius, cos(self.pressure_angle)))

    def gen_gear_tooth(self) -> PointList:
        r"""
            Generate one tooth centered on the tip of the tooth.
            Does not include the root flats since they will be created when adjoining
            tooth is placed.
                  ____
                 /    \
            \___/      \___/
        """
        addendum = self.module
        dedendum = self.module * self.relief_factor
        tooth = self.pitch / 2
        addendum_offset = addendum * tan(self.pressure_angle)
        dedendum_offset = dedendum * tan(self.pressure_angle)
        # print(self.pitch, tooth, addendum_offset)
        # print('pitch=', self.pitch, ' cp=', tau / self.teeth)
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

        start_angle = -tooth_offset / pr
        pts = inv.path(offset=start_angle, center=self.center, up=-1, steps=self.steps)
        # TODO-fix all of this reversing
        points.extend(reversed(pts))
        if not inv.clipped:
            # Add the direct line to the dedendum radius
            # TODO-this should handle undercutting
            points.append((dr * cos(start_angle) + cx, dr * sin(start_angle) + cy))
        points.extend(Point(x, -y) for x, y in list(reversed(points)))
        points = [Point(x, y) for x, y in reversed(points)]
        return points

    def gen_by_rack(self):
        """Generate the gear shape by moving a rack past the gear"""
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle),
                    relief_factor=self.relief_factor, tall_tooth=True)
        gear_points = []
        steps = 50
        z_teeth = 3
        rack_x = self.pitch_radius + self.center.x
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        only_one_edge = False
        if only_one_edge:
            tooth_pts = tooth_pts[:2]
        for step in range(-steps, steps+1):
            tooth_pos = z_teeth * step / steps
            rack_y = tooth_pos * self.pitch + self.center.y
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
        assert(self.center == Point(0, 0))
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle),
                    relief_factor=self.relief_factor, tall_tooth=not True)
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
        assert(self.center == Point(0, 0))

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

    def gen_rack_tooth(self):
        rack = Rack(module=self.module, pressure_angle=degrees(self.pressure_angle), relief_factor=self.relief_factor)
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        offset = Vector(self.pitch_radius, 0) + Vector(*self.center)
        return [(p+offset).xy() for p in tooth_pts]

    def plot(self, color='red', tool_angle=40.0, gear_space=None, mill_space=None):
        addendum = self.module
        dedendum = self.module * self.relief_factor
        pitch_radius = self.pitch_radius
        if mill_space is None and gear_space is None:
            gear_space = True

        if self.pressure_line:
            dx = self.module*5*sin(self.pressure_angle)
            dy = self.module*5*cos(self.pressure_angle)
            cx = pitch_radius + self.center.x
            cy = 0 + self.center.y
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
            plot(self.gen_rack_tooth(), 'green')

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

        plot(self.instance().poly, color=color)
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

    def instance(self, x_pos=0):
        """Return a gear instance that represents this gear"""
        # x_pos = wheel_pitch_radius + pinion_pitch_radius
        return GearInstance(self.module, self.teeth, 'Involute', '', self.gen_gear_tooth(), Point(x_pos, 0),
                            tip_radius=self.tip_radius, base_radius=self.base_radius,
                            root_radius=self.dedendum_radius)


class InvolutePair:
    """Wheel and Pinion involute gears"""
    def __init__(self, wheel_teeth=30, pinion_teeth=6,
                 module=1.0, relief_factor=1.25,
                 pressure_angle=20.0):
        self.gear_wheel = GearInvolute(teeth=wheel_teeth, module=module, relief_factor=relief_factor, pressure_angle=pressure_angle)
        self.gear_pinion = GearInvolute(teeth=pinion_teeth, module=module, relief_factor=relief_factor, pressure_angle=pressure_angle)

    def wheel(self):
        return self.gear_wheel.instance()

    def pinion(self):
        p = self.gear_pinion.instance()
        p.center = Point(self.gear_wheel.pitch_radius + self.gear_pinion.pitch_radius, 0)
        p.rotation_extra = 0.5 if p.teeth % 2 == 0 else 0.0
        return p

    def plot(self, color='blue', rotation=0.0, plotter=None):
        self.wheel().plot(color, rotation, plotter)
        self.pinion().plot(color, rotation, plotter)
