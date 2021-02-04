import numpy as np
import scipy.optimize
from math import sqrt, cos, sin, pi, radians, tan, atan2, degrees, hypot, tau
from numbers import Number
from typing import List, Tuple, Union, Any

from matplotlib import pyplot as plt

from x7.lib.iters import t_range
from x7.geom.geom import Point, Vector, Line, PointList
from x7.geom.transform import Transform
from x7.geom.utils import circle, path_to_xy
from x7.geom.plot_utils import plot
from gear_base import GearInstance
from rack import Rack


class InvoluteWithOffsets(object):
    """
        Involute curve, with added x & y offsets to be more usable for gears.

        See https://en.wikipedia.org/wiki/Involute#Involutes_of_a_circle
    """

    def __init__(self, radius=0.0, offset_angle=0.0, offset_radius=0.0, offset_norm=0.0,
                 radius_min=0.0, radius_max=0.0):
        """
            :param radius:          Base radius for gear involutes, pitch radius for gear trochoids
            :param offset_angle:    Starting rotation of curve
            :param offset_radius:   Offset of point into circle (taller rack)
            :param offset_norm:     Offset of point perpendicular (CCW) to radius (wider rack tooth)
            :param radius_min:      Minimum radius (== 0 implies calculated minimum)
            :param radius_max:      Maximum radius (== 0 implies no maximum)
        """

        self.radius = radius
        self.offset_angle = offset_angle
        self.offset_radius = offset_radius
        self.offset_norm = offset_norm
        if radius_min:
            if radius_min < radius-offset_radius:
                raise ValueError('radius_min')
            self.radius_min = radius_min
        else:
            self.radius_min = radius-offset_radius
        self.radius_max = radius_max
        self.start_angle = self.calc_angle(self.radius_min)
        self.end_angle = self.calc_angle(self.radius_max) if radius_max else (2*tau)
        debug_print = False
        if debug_print:
            print('InvWO: r=%.6f mnr=%.6f mxr=%.6f sa=%.6f ea=%.6f' %
                  (self.radius, self.radius_min, self.radius_max, self.start_angle, self.end_angle))

    def mid_angle(self):
        """Average of .start_angle and .end_angle"""
        return (self.start_angle + self.end_angle) / 2

    def calc_point(self, angle, clip=True):
        """Involute with offset_radius and offset_norm (which makes this a generalized trochoid)"""
        # x = r * cos(a) + r*(a-oa) * sin(a)
        # x = (r-or) * cos(a) + r*(a-oa-on) * sin(a)
        angle += self.offset_angle
        x = (self.radius-self.offset_radius) * cos(angle) + (self.radius*(angle - self.offset_angle)-self.offset_norm) * sin(angle)
        y = (self.radius-self.offset_radius) * sin(angle) - (self.radius*(angle - self.offset_angle)-self.offset_norm) * cos(angle)
        # y = self.radius * (sin(angle) - (angle - offset) * cos(angle))
        return x, y

    def calc_angle(self, distance) -> float:
        """Calculate angle (radians) for corresponding distance (> radius)"""
        if self.offset_norm or self.offset_radius:
            # Need to calculate numerically
            def objective(t):
                x, y = self.calc_point(t[0], False)
                return np.array([distance - hypot(x, y)])
            solution, info, ok, msg = scipy.optimize.fsolve(objective, np.array([0]), full_output=True)
            if not ok:
                raise ValueError(msg)
            return solution[0]
        else:
            # Simple closed form exists for plain involute
            assert distance >= self.radius
            return sqrt(distance * distance / (self.radius * self.radius) - 1)

    def x_calc_undercut_t_at_r(self, r):
        # r = rp*sqrt(sqr(1-2*relief/N) + sqr(pi/(2*N) - 2*relief/N*tan(pa)-t))
        # solve rp*sqrt((1-2*relief/N)^2 + (pi/(2*N) - 2*relief/N*tan(pa)-t)^2) - r = 0 for t
        # solve p*sqrt((1-2*k/N)^2 + (pi/(2*N) - 2*k/N*tan(a)-t)^2) = r for t
        # t = 1/2 (-(4 k tan(a))/N + (2 sqrt(-4 k^2 p^2 + 4 k N p^2 - N^2 p^2 + N^2 r^2))/(N p) + π/N)
        pass

    def path(self, steps=10, offset=0.0, up=1, clip=False) -> List[Tuple[float, float]]:
        """Generate path for involute"""
        t_vals = t_range(steps, self.start_angle, self.end_angle)
        points = (self.calc_point(t) for t in t_vals)
        # Sigh.  Pycharm thinks filter(None, ...) is Iterator[None]
        if clip:
            points = ((x, y) for x, y in points if self.radius_min <= hypot(x, y) <= self.radius_max)
        return list(points)


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

    def calc_undercut_t_at_r(self, r):
        # r = rp*sqrt(sqr(1-2*relief/N) + sqr(pi/(2*N) - 2*relief/N*tan(pa)-t))
        # solve rp*sqrt((1-2*relief/N)^2 + (pi/(2*N) - 2*relief/N*tan(pa)-t)^2) - r = 0 for t
        # solve p*sqrt((1-2*k/N)^2 + (pi/(2*N) - 2*k/N*tan(a)-t)^2) = r for t
        # t = 1/2 (-(4 k tan(a))/N + (2 sqrt(-4 k^2 p^2 + 4 k N p^2 - N^2 p^2 + N^2 r^2))/(N p) + π/N)
        pass

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

    def path(self, steps=10, offset=0.0, up=1, center=(0.0, 0.0)) -> List[Tuple[float, float]]:
        """Generate path for involute"""
        factor = up * (self.end_angle - self.start_angle) / steps
        sa = up * self.start_angle
        zp = (self.calc_point(factor * step + offset + sa, offset=offset) for step in range(0, steps + 1))
        cx, cy = center
        return [(x + cx, y + cy) for x, y in zp]


class GearInvolute(object):
    HIGH_QUALITY = dict(steps=20, tip_arc=0.5, root_arc=0.5, curved_root=True)

    def __init__(self, teeth=30, center=Point(0, 0), rot=0.0,
                 module=1.0, relief_factor=1.25,
                 steps=4, tip_arc=0.0, root_arc=0.0, curved_root=False, debug=False,
                 pressure_angle=20.0, profile_shift=0.0):
        """
            Plot an involute gear
            :param teeth:	Number of teeth in gear
            :param center:  Center of gear
            :param rot: 	Rotation in #teeth
            :param module:	Module of gear
            :param relief_factor: Relief factor
            :param steps:   Number of steps on the gear face (and the undercut)
            :param tip_arc: Max degrees per step for arc at tip of tooth (0==no arc, just straight line)
            :param root_arc: Max degrees per step for arc at root of tooth (0==no arc, just straight line)
            :param curved_root: True to include root curve even if no undercut
            :param debug: True for verbose messages and plotting during construction
            :param pressure_angle: Pressure angle
            :param profile_shift: Profile shift as fraction of module
        """
        self.teeth = teeth
        self.module = module
        self.center = center
        self.rot = rot
        self.relief_factor = relief_factor
        self.steps = steps
        self.tip_arc = tip_arc
        self.root_arc = root_arc
        self.curved_root = curved_root
        self.debug = debug
        self.pressure_angle = radians(pressure_angle)
        self.profile_shift = profile_shift

    @property
    def pitch(self):
        return self.module * pi

    @property
    def pitch_radius(self):
        return self.module * self.teeth / 2

    @property
    def pitch_radius_effective(self):
        """Pitch radius with profile_shift considered"""
        return self.pitch_radius + self.module * self.profile_shift

    @property
    def base_radius(self):
        return self.pitch_radius * cos(self.pressure_angle)

    @property
    def tip_radius(self):
        return self.pitch_radius + self.module * (1 + self.profile_shift)

    @property
    def root_radius(self):
        return self.pitch_radius - self.module * (self.relief_factor - self.profile_shift)

    def copy(self):
        """Deep copy"""
        return type(self)(
            teeth=self.teeth,
            center=self.center.copy(),
            rot=self.rot,
            module=self.module,
            relief_factor=self.relief_factor,
            steps=self.steps,
            tip_arc=self.tip_arc,
            root_arc=self.root_arc,
            curved_root=self.curved_root,
            debug=self.debug,
            pressure_angle=degrees(self.pressure_angle),
            profile_shift=self.profile_shift,
        )

    def restore(self, other: 'GearInvolute'):
        """Restore in place"""
        self.teeth = other.teeth
        self.center = other.center.copy()
        self.rot = other.rot
        self.module = other.module
        self.relief_factor = other.relief_factor
        self.steps = other.steps
        self.tip_arc = other.tip_arc
        self.root_arc = other.root_arc
        self.curved_root = other.curved_root
        self.debug = other.debug
        self.pressure_angle = other.pressure_angle
        self.profile_shift = other.profile_shift

    def __eq__(self, other):
        return (self.teeth == other.teeth
                and self.center == other.center
                and self.rot == other.rot
                and self.module == other.module
                and self.relief_factor == other.relief_factor
                and self.steps == other.steps
                and self.tip_arc == other.tip_arc
                and self.root_arc == other.root_arc
                and self.curved_root == other.curved_root
                and self.debug == other.debug
                and self.pressure_angle == other.pressure_angle
                and self.profile_shift == other.profile_shift
                )

    @property
    def pressure_angle_degrees(self):
        return degrees(self.pressure_angle)

    @pressure_angle_degrees.setter
    def pressure_angle_degrees(self, pa):
        self.pressure_angle = radians(pa)

    @property
    def pressure_angle_radians(self):
        return self.pressure_angle

    @pressure_angle_radians.setter
    def pressure_angle_radians(self, pa):
        self.pressure_angle = pa

    def min_teeth_without_undercut(self):
        # TODO-include profile_shift
        sin_pa = sin(self.pressure_angle)
        return 2 / (sin_pa * sin_pa)

    def _finish_tooth_parts(self, parts, root_radius=0.0, closed=False):
        root_radius = root_radius or self.root_radius
        other_side = [(tag, [(x, -y) for x, y in reversed(points)]) for tag, points in reversed(parts)]
        cs = 1 if closed else 0

        if self.tip_arc:
            last_point = parts[-1][-1][-1]
            tip_angle = abs(atan2(last_point[1], last_point[0]))
            tip_angle_degrees = degrees(tip_angle * 2)
            if tip_angle_degrees > self.tip_arc:
                steps = int(tip_angle_degrees / self.tip_arc) + 1
                if self.debug:
                    print('total tip angle=%.2f  steps=%d' % (tip_angle_degrees, steps))
                tip_arc = []
                for n in range(1-cs, steps+cs):
                    t = n / steps * tip_angle * 2 - tip_angle
                    tip_arc.append((self.tip_radius * cos(t), self.tip_radius * sin(t)))
                parts.append(('tip_arc', tip_arc))
        parts.extend(other_side)
        if self.root_arc:
            last_point = parts[-1][-1][-1]
            root_angle = abs(pi / self.teeth - (atan2(last_point[1], last_point[0])))
            root_angle_degrees = degrees(root_angle * 2)
            if root_angle_degrees > self.root_arc:
                steps = int(root_angle_degrees / self.root_arc) + 1
                if self.debug:
                    print('total root angle=%.2f  steps=%d' % (root_angle_degrees, steps))
                root_arc = []
                for n in range(1-cs, steps+cs):
                    t = n / steps * root_angle * 2 - root_angle + pi / self.teeth
                    root_arc.append((root_radius * cos(t), root_radius * sin(t)))
                parts.append(('root_arc', root_arc))

    def gen_gear_tooth(self) -> PointList:
        """Convert gen_gear_tooth_parts result into plain list of points"""
        return [p for tag, points in self.gen_gear_tooth_parts() for p in points if tag[0] != '_']

    def gen_gear_tooth_parts(self, closed=False, include_extras=False) -> List[Tuple[str, Union[PointList, Any]]]:
        r"""
            Generate one tooth centered on the tip of the tooth.
            Does not include the root flats since they will be created when adjoining
            tooth is placed.
                  ____
                 /    \
            \___/      \___/


            :param closed: True to include duplicate points and make each part stand-alone plottable
            :param include_extras: True to include extra items in parts with '_' tags
            :return: List of (part-name, part-path)
        """
        ps = self.profile_shift * self.module
        addendum_ps = self.module - ps
        dedendum_ps = self.module * self.relief_factor - ps
        tooth = self.pitch / 2
        half_tooth = tooth / 2
        addendum_offset = half_tooth - (addendum_ps+ps) * tan(self.pressure_angle)
        dedendum_offset = half_tooth - (dedendum_ps+ps) * tan(self.pressure_angle)

        pr = self.pitch_radius

        gear_face = InvoluteWithOffsets(self.base_radius, radius_max=self.tip_radius,
                                        radius_min=max(self.base_radius, self.root_radius))
        # Calc pitch point where involute intersects pitch circle and offset
        pp_inv_angle = gear_face.calc_angle(self.pitch_radius)
        ppx, ppy = gear_face.calc_point(pp_inv_angle)
        pp_off_angle = atan2(ppy, ppx)

        # Calculate expected gap width at pitch_radius including profile_shift
        gap_width = tooth / 2 - ps * tan(self.pressure_angle_radians)

        # Multiply pp_off_angle by pr to move from angular to pitch space
        tooth_offset = gap_width - pp_off_angle * pr
        # tooth_offset = tooth / 2 - pp_off_angle * pr
        gear_face.offset_angle = (tooth_offset - tooth) / pr

        parts = []
        extras = []

        undercut_required = self.teeth < self.min_teeth_without_undercut()
        if undercut_required or self.curved_root:
            short_tip = False
            if short_tip:       # Use just addendum to generate undercut (mostly for plotting)
                undercut = InvoluteWithOffsets(self.pitch_radius, offset_angle=-tooth/pr,
                                               offset_radius=addendum_ps, offset_norm=addendum_offset,
                                               radius_min=0, radius_max=self.tip_radius)
            else:
                undercut = InvoluteWithOffsets(self.pitch_radius, offset_angle=-tooth/pr,
                                               offset_radius=dedendum_ps, offset_norm=dedendum_offset,
                                               radius_min=0, radius_max=self.tip_radius)

            # Find the intersection of gear_face and undercut
            def objective(ab):
                a, b = ab
                ax, ay = gear_face.calc_point(a)
                bx, by = undercut.calc_point(b)
                return np.array([ax - bx, ay - by])

            guesses = np.array([gear_face.mid_angle(), undercut.mid_angle()])
            result, info, ok, message = scipy.optimize.fsolve(objective, guesses, full_output=True)
            if not ok:
                if self.debug:
                    from pprint import pp
                    pp((result, info, ok, message))
                raise ValueError('Undercut / Face intersection: %s' % message)

            gear_face.start_angle = result[0]
            undercut.end_angle = result[1]

            if closed:
                parts.append(('root_cut', undercut.path(self.steps)))
            else:
                parts.append(('root_cut', undercut.path(self.steps)[:-1]))
            parts.append(('face', gear_face.path(self.steps)))

            self._finish_tooth_parts(parts, root_radius=undercut.radius - undercut.offset_radius, closed=closed)

            if include_extras:
                intersection = Point(*gear_face.calc_point(result[0]))
                extras.append(('_face_root_intersection', intersection))
                extras.append(('_root_cut', undercut))
        else:
            face_path = gear_face.path(self.steps)
            if self.base_radius > self.root_radius:
                dropcut = [(Vector(*face_path[0]).unit() * self.root_radius).xy()]
                if closed:
                    dropcut.append(face_path[0])
                parts.append(('dropcut', dropcut))
            parts.append(('face', face_path))
            self._finish_tooth_parts(parts, closed=closed)

        if include_extras:
            extras.extend([
                ('_gear_face', gear_face),
                ('_locals', locals()),
            ])

        parts = [(tag, [Point(x, y) for x, y in reversed(points)]) for tag, points in reversed(parts)]
        return parts + extras

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

    def rack(self) -> Rack:
        return Rack(module=self.module, pressure_angle=self.pressure_angle_degrees, relief_factor=self.relief_factor)

    def gen_rack_tooth(self, teeth=1, rot=0.5, as_pt=False):
        """Generate a rack with teeth (must be odd)"""
        rack = self.rack()
        tooth_pts = [rack.tooth_base_high, rack.tooth_tip_high, rack.tooth_tip_low, rack.tooth_base_low]
        y_shift = Vector(0, -self.pitch)
        offset = Vector(self.pitch_radius_effective, 0) + Vector(*self.center) + y_shift * (rot % 1)
        teeth //= 2
        path = [(p+offset+y_shift*tooth) for tooth in range(-teeth, teeth+1) for p in tooth_pts]
        return path if as_pt else path_to_xy(path)

    def plot(self, color='red', tool_angle=40.0,
             gear_space=None, mill_space=None,
             pressure_line=True, linewidth=1) -> str:
        """Plot the gear, along with construction lines & circles.  Returns additional text to display"""

        addendum = self.module
        pitch_radius = self.pitch_radius
        if mill_space is None and gear_space is None:
            gear_space = True

        if pressure_line:
            dx = self.module*5*sin(self.pressure_angle)
            dy = self.module*5*cos(self.pressure_angle)
            cx = pitch_radius + self.center.x
            cy = 0 + self.center.y
            plot([(cx+dx, cy+dy), (cx-dx, cy-dy)], color='#FFA0FF', label='Pressure Line')
            if pressure_line > 1:
                plot([(cx-dx, cy+dy), (cx+dx, cy-dy)], color='#FFA0FF')

        extra_text = '\n'.join([
            'Module: $m$',
            'Number of Teeth: $z$',
            'Profile Shift: $x$',
            'Pressure Angle: $\\alpha$'])
        pr_label = r'Pitch Radius: $r_p = \frac{m \cdot z}{2}$'
        if self.pitch_radius != self.pitch_radius_effective:
            plot(circle(pitch_radius, c=self.center), color='green', linestyle=':', label=pr_label)
            plot(circle(self.pitch_radius_effective, c=self.center), color='green', label='Pitch Radius Effective: $r_e= r_p + x$')
        else:
            plot(circle(pitch_radius, c=self.center), color='green', label=pr_label)
        plot(circle(self.tip_radius, c=self.center), color='yellow', label='Tip Radius: $r_t = r_p + H_a$')
        plot(circle(self.root_radius, c=self.center), color='blue', label='Root Radius: $r_r = r_p - H_d$')
        plot(circle(self.pitch_radius_effective - addendum, c=self.center), color='cyan', label='Max Tip Depth: $td_{max} = r_e - H_a$')
        plot(circle(self.base_radius, c=self.center), color='orange', label=r'Base Radius: $r_b = r_p \cos \alpha$')

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

        poly = self.instance().poly_at()
        poly.append(poly[0])
        plot(poly, color=color, linewidth=linewidth)
        # self.gen_gcode()
        return extra_text

    def plot_show(self, zoom_radius: Union[None, float, Tuple[float, float, float]] = None,
                  axis='x', grid=True):
        plt.axis('equal')
        if grid:
            plt.grid()
        # Set zoom_radius to zoom in around where gears meet
        if zoom_radius:
            zxl, zxr, zy = zoom_radius if isinstance(zoom_radius, tuple) else (zoom_radius, zoom_radius, zoom_radius)
            ax = plt.gca()
            if axis in ('x', '-x'):
                pre = self.pitch_radius_effective if axis == 'x' else -self.pitch_radius_effective
                ax.set_xlim(pre - zxl, pre + zxr)
                ax.set_ylim(-zy, zy)
            else:
                pre = self.pitch_radius_effective if axis == 'y' else -self.pitch_radius_effective
                ax.set_ylim(pre - zxl, pre + zxr)
                ax.set_xlim(-zy, zy)
        plt.show()

    def instance(self):
        """
            Return a gear instance that represents this gear
            :return: GearInstance
        """
        """"""
        # x_pos = wheel_pitch_radius + pinion_pitch_radius

        return GearInstance(self.module, self.teeth, 'Involute', '', self.gen_gear_tooth(),
                            center=self.center, rotation_extra=self.rot,
                            tip_radius=self.tip_radius, base_radius=self.base_radius,
                            root_radius=self.root_radius)


class InvolutePair:
    """Wheel and Pinion involute gears"""
    def __init__(self, wheel_teeth=30, pinion_teeth=6,
                 module=1.0, relief_factor=1.25,
                 steps=4, tip_arc=0.0, root_arc=0.0, curved_root=False, debug=False,
                 pressure_angle=20.0):
        self.gear_wheel = GearInvolute(teeth=wheel_teeth, module=module,
                                       relief_factor=relief_factor, pressure_angle=pressure_angle,
                                       steps=steps, tip_arc=tip_arc, root_arc=root_arc, curved_root=curved_root,
                                       debug=debug)
        self.gear_pinion = GearInvolute(teeth=pinion_teeth, module=module,
                                        relief_factor=relief_factor, pressure_angle=pressure_angle,
                                        steps=steps, tip_arc=tip_arc, root_arc=root_arc, curved_root=curved_root,
                                        debug=debug)

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


def main():
    nt = 14
    # for rot in [0, 0.25, 0.5]:
    # for rot in t_range(10, 0, 0.5):
    # for rot in [0, 0.5, 1, 3, 5, nt]:
    for rot in [0]:
        gi = GearInvolute(nt, rot=rot, module=3, **GearInvolute.HIGH_QUALITY)
        g = gi.instance()
        # g.plot('c')
        plot(g.tooth_path, 'b:')
        for ps in [0, 0.5]:
            gi2 = GearInvolute(nt, rot=rot, module=g.module, profile_shift=ps, center=Point(0.0, 0), **GearInvolute.HIGH_QUALITY)
            g2 = gi2.instance()
            g2.plot('g')
            show_rack = True
            if show_rack:
                rack = GearInvolute(129, rot=-rot, module=g.module, profile_shift=0, **GearInvolute.HIGH_QUALITY)
                rack.center = Point(rack.pitch_radius_effective + gi2.pitch_radius_effective, 0.0)
                rack.instance().plot()
        g.plot_show(5*g.module)


if __name__ == '__main__':
    main()
