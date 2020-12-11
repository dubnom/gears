from math import sin, cos, tau, radians, ceil, tan
from typing import List, Union, Tuple, Optional

from matplotlib import pyplot as plt

from anim.geom import BasePoint, Point, Vector, XYList, Line, polygon_area, iter_rotate
from anim.transform import Transform

__all__ = [
    'PointList', 'PointUnionList', 'check_point_list',
    't_range',
    'path_rotate', 'path_translate', 'circle', 'arc', 'plot',
    'GearInstance', 'CUT_KIND_COLOR_MAP', 'CutError',
]
PointList = List[BasePoint]
PointUnionList = List[Union[BasePoint, float, Tuple[float, float]]]
CUT_KIND_COLOR_MAP = {
    'undercut': 'red',
    'ascending': '#0000FF',
    'descending': '#00FF00',
    'out': '#9999FF',
    'in': '#99FF99',
    'flat': 'orange',
    'flat-tip': 'orange',
    'flat-root': 'orange',
}
CUT_KINDS = set(CUT_KIND_COLOR_MAP.keys())


class CutError(Exception):
    pass


def check_point_list(lst):
    """Validate that lst is List[BasePoint]"""
    for elem in lst:
        if not isinstance(elem, BasePoint):
            raise ValueError('%s of lst is not a BasePoint' % repr(elem))


def t_range(steps, t_low=0.0, t_high=1.0, closed=True):
    """Range from t_low to t_high in steps.  If closed, include t_high"""
    t_len = t_high - t_low
    return (step / steps * t_len + t_low for step in range(0, steps + int(closed)))


def path_rotate(path: PointList, angle, as_pt=False) -> PointUnionList:
    """Rotate all points by angle (in degrees) about 0,0"""
    t = Transform().rotate(angle)
    return [t.transform_pt(pt, as_pt) for pt in path]


def path_translate(path: PointList, dxy: Union[Point, Vector], as_pt=False) -> PointUnionList:
    """Rotate all points by angle (in degrees) about 0,0"""
    dxy = Vector(*dxy.xy())
    if as_pt:
        return [p + dxy for p in path]
    else:
        return [(p + dxy).xy() for p in path]


def arc(r, sa, ea, c=Point(0, 0)) -> XYList:
    """Generate an arc of radius r about c as a list of x,y pairs"""
    steps = int(abs(ea-sa)+1)
    return [(r * cos(t) + c.x, r * sin(t) + c.y) for t in t_range(steps, radians(sa), radians(ea))]


def circle(r, c=Point(0, 0)) -> XYList:
    """Generate a circle of radius r about c as a list of x,y pairs"""
    steps = 360
    return [(r * cos(t) + c.x, r * sin(t) + c.y) for t in t_range(steps, 0, tau)]


def plot(xy, color='black', plotter=None):
    plotter = plotter or plt
    plotter.plot(*zip(*xy), color)


class CutDetail:
    """
        Detail information for generating GCODE
    """
    def __init__(self, line: Line, kind: str, angle: float, y: float, z: float):
        self.line = line
        self.kind = kind
        self.angle = angle
        self.y = y
        self.z = z

    def __str__(self):
        return 'CutDetail(%s, %s, %.4f, %.4f, %.4f)' % (self.line, self.kind, self.angle, self.y, self.z)


class ClassifiedCut:
    """
        A cut classified by direction
    """
    def __init__(self, line: Line, kind: str,
                 convex_p1: bool, convex_p2: bool, overshoot: float, z_offset: float):
        """
            :param line:        Line segment in CCW direction
            :param kind:        in/out/ascending/descending/flat-tip/flat-root/undercut
            :param convex_p1:      True if shape is convex at line.p1
            :param convex_p2:      True if shape is convex at line.p2
            :param overshoot:   Allowed overshoot
            :param z_offset:    Z offset for tool
        """
        self.line = line
        assert kind in CUT_KINDS
        self.kind = kind
        self.convex_p1 = convex_p1
        self.convex_p2 = convex_p2
        self.overshoot = overshoot
        self.z_offset = z_offset
        # cut_line is reversed if this is an inward cut
        # cut_line.origin is always the deepest point of cut
        self.cut_line = Line(line.p2, line.p1) if self.inward() else self.line
        self.cut_details: List[CutDetail] = []

    def __str__(self):
        c1 = 'convex1' if self.convex_p1 else 'concave1'
        c2 = 'convex2' if self.convex_p2 else 'concave2'
        return '%-10s %-8s %-8s %8.3f %8.3f %s' % (self.kind, c1, c2, self.overshoot, self.radius(), self.cut_line)

    def radius(self):
        """Radius of the deepest point of this cut"""
        return Vector(*self.cut_line.origin.xy()).length()

    def flat(self):
        """True if kind is 'flat-tip' or 'flat-root'"""
        return self.kind in {'flat-tip', 'flat-root'}

    def inward(self):
        """True if kind is 'in' or 'descending'"""
        return self.kind in {'in', 'descending'}

    def outward(self):
        """True if kind is 'out' or 'ascending'"""
        return self.kind in {'out', 'ascending'}


class GearInstance:
    """Holder for finished gear (could be just about any polygon, really)"""
    def __init__(self, module, teeth, shape, kind, tooth_path: PointList, center: Point,
                 poly: Optional[PointList] = None,
                 tip_radius=0., base_radius=0., root_radius=0.):
        self.module = module
        self.teeth = teeth
        self.shape = shape
        self.kind = kind
        self.tooth_path = tooth_path
        check_point_list(tooth_path)
        self.poly = poly or self.gen_poly()
        check_point_list(self.poly)
        self.center = center
        # self.circular_pitch = self.module * pi
        self.pitch_diameter = self.module * self.teeth
        self.pitch_radius = self.pitch_diameter / 2
        self.tip_radius = tip_radius
        self.base_radius = base_radius
        self.root_radius = root_radius

    def gen_poly(self, rotation=0.0) -> PointList:
        """Generate the full polygon from the tooth path"""
        degrees_per_tooth = 360 / self.teeth
        rotation *= degrees_per_tooth

        points = []
        for n in range(self.teeth):
            mid = n * degrees_per_tooth + rotation
            points.extend(path_rotate(self.tooth_path, mid, True))
        check_point_list(points)
        return points

    def description(self):
        return '%s %s%d teeth' % (self.shape, (self.kind + ' ' if self.kind else ''), self.teeth)

    def __str__(self):
        return 'GearInstance: %s %s points @ %s' % (self.description(), len(self.poly), self.center.round(2))

    def plot(self, color='blue', rotation=0.0, plotter=None):
        """
            Plot the gear poly and associated construction lines/circles

            :param color:       color for gear outline
            :param rotation:    in units of teeth
            :param plotter:     matplotlib Axes or None for top-level plot
            :return:
        """
        rotation *= 360 / self.teeth
        plot(circle(self.pitch_radius, self.center), 'lightgreen', plotter=plotter)
        if self.base_radius:
            plot(circle(self.base_radius, self.center), 'wheat', plotter=plotter)
        if self.tip_radius:
            plot(circle(self.tip_radius, self.center), 'yellow', plotter=plotter)
        if self.root_radius:
            plot(circle(self.root_radius, self.center), 'yellow', plotter=plotter)
        path = path_translate(path_rotate(self.poly, rotation, True), self.center)
        path.append(path[0])        # Make sure it is closed
        plot(path, color, plotter=plotter)
        plot(path_translate(path_rotate([Point(0, 0), self.poly[5*len(self.poly)//self.teeth]], rotation, True), self.center), color, plotter=plotter)

    def set_zoom(self, zoom_radius=0.0, plotter=None):
        plotter = plotter or plt
        plotter.axis('equal')
        plotter.grid()
        # Set zoom_radius to zoom in around where gears meet
        if zoom_radius:
            from matplotlib.axes import Axes
            ax = plotter if isinstance(plotter, Axes) else plt.gca()
            ax.set_xlim(self.pitch_radius - zoom_radius, self.pitch_radius + zoom_radius)
            ax.set_ylim(-zoom_radius, zoom_radius)

    def plot_show(self, zoom_radius=0.0, plotter=None):
        self.set_zoom(zoom_radius=zoom_radius, plotter=plotter)
        plt.show()

    def classify_cuts_pass1(self, tool_angle, tool_tip_height=0.0) -> List[ClassifiedCut]:
        """
            Classify the cuts in this polygon, first pass.  No error checking of cuts in this pass.

            :param tool_angle: tool angle in degrees
            :param tool_tip_height:
            :returns: List of ClassifiedCut

            Overshoot is allowed if the cut is along a convex part of the polygon.

            Classify cuts by angle relative to center of gear::

               <--- iterating CCW around top of gear
                              out
                       ending  |    u
                     c         |       n
                    s          |         d
                    a          |          e
               flat ---------- + ---------r-
                    d          |
                    e          |         c
                     s         |       u
                      cending  |    t
                               in

        """
        # Is polygon in CCW order?
        poly = self.poly
        if polygon_area(poly) < 0:
            poly = list(reversed(poly))
        per_tooth = len(poly) // self.teeth
        poly = poly[-per_tooth:] + poly[:-per_tooth]
        half_tool_tip = tool_tip_height / 2

        cuts = []

        vertical_eps = 1.0  # degrees
        flat_eps = 15.0     # degrees
        for a, b, c in iter_rotate(poly, 3):
            cut = Line(a, b)
            # TODO-decide whether to take angle from tip or center of cut
            angle_from_midpoint = True
            if angle_from_midpoint:
                radial_vector = Vector(*a.mid(b).xy())
            else:
                # TODO-not quite the tip, since it's not clear if a or b more inward yet
                radial_vector = Vector(*a.xy())
            radial_angle = radial_vector.angle()
            radial_distance = radial_vector.length()
            line_angle = cut.direction.angle()
            delta = line_angle - radial_angle
            if delta > 180:
                delta -= 360
            if delta < -180:
                delta += 360
            # print('a:%-8s b:%-8s ra:%5.1f la:%5.1f d:%5.1f' % (a, b, radial_angle, line_angle, delta))
            if abs(delta) < vertical_eps:
                kind = 'out'
            elif 180 - abs(delta) < vertical_eps:
                kind = 'in'
            elif abs(delta-90) < flat_eps:
                if radial_distance < self.pitch_radius:
                    kind = 'flat-root'
                else:
                    kind = 'flat-tip'
            elif 90 <= delta < 180:
                kind = 'descending'
            elif 0 < delta <= 90:
                kind = 'ascending'
            else:
                kind = 'undercut'
            convex_p1 = cuts[-1].convex_p2 if cuts else False
            convex_p2 = cut.direction.cross(c-b) >= 0
            # TODO-overshoot should look for intersections with other segments, based on tool shape
            if kind in {'ascending', 'out'}:
                # On ascending or out cuts, we care about the previous intersection's convexity, not the trailing
                overshoot = 1.0 if convex_p1 else 0.0
                z_offset = half_tool_tip
            else:
                overshoot = 1.0 if convex_p2 else 0.0
                z_offset = -half_tool_tip

            cuts.append(ClassifiedCut(
                line=cut, kind=kind, convex_p1=convex_p1, convex_p2=convex_p2,
                overshoot=overshoot, z_offset=z_offset))

        cuts[0].convex_p1 = cuts[-1].convex_p2

        debug_print = False
        if debug_print:
            for cut in cuts[:20]:
                print(cut)
        return cuts

    def classify_cuts_pass2(self, classified: List[ClassifiedCut], tool_angle, tool_tip_height=0.0) -> List[ClassifiedCut]:
        """
            Refine the classified cuts:
            * error checking
            * reorient the cut.cut_line to tool alignment

            :param classified: classified cuts from pass1
            :param tool_angle: tool angle in degrees
            :param tool_tip_height:
            :returns: List of (cut, cut-kind, convex, allowed overshoot)
        """
        half_tool_tip = tool_tip_height / 2
        if tool_angle:
            # For pointy tools clearing the flats at the bottom of the tooth,
            # go deeper with the cutter by _extension which gives a width of _tip
            # TODO-0.3 seems to work, but should really be configurable
            flat_tool_extension = 0.3 * self.module
            flat_tool_tip = flat_tool_extension * tan(radians(tool_angle/2)) * 2
        else:
            flat_tool_extension = 0
            flat_tool_tip = tool_tip_height

        # Rotate classified cuts until we find a the first edge after an "in" edge
        orig_len = len(classified)
        last_in = -1
        while not classified[last_in-1].inward():
            last_in -= 1
        if last_in != -1:
            classified = classified[last_in+1:] + classified[:last_in+1]
        assert len(classified) == orig_len

        for cut_index, cut in enumerate(classified):
            cuts = []
            is_flat = (cut.kind == 'flat-root') or (cut.kind == 'flat-tip' and tool_angle == 0)
            if cut.kind == 'undercut':
                raise ValueError("Can't do undercuts yet")
            elif is_flat:
                if tool_angle != 0 and Vector(*cut.cut_line.origin.xy()).length() < self.pitch_radius:
                    # TODO-attempt to do bottom clearing with pointy-cutter
                    pass
                    # continue
                if tool_angle == 0:
                    normal = cut.cut_line.direction.unit().normal() * cut.z_offset
                else:
                    normal = cut.cut_line.direction.unit().normal() * -1
                du = cut.line.direction.unit() * flat_tool_tip
                length = cut.line.direction.length()
                if length == 0:
                    continue
                elif cut.line.direction.length() < flat_tool_tip:
                    # Cut shorter than saw kerf
                    if not cut.convex_p1 and not cut.convex_p2:
                        msg = 'ERROR: Cut is too narrow for tool.  Cut: %s [%.6f len]  Tool tip: %.4f' \
                              % (cut, length, flat_tool_tip)
                        print(msg)
                        raise CutError(msg)
                    if not cut.convex_p1:
                        # Align cut to p1
                        cuts.append((Line(cut.line.p1 + du, -1 * normal), 'flat-p1'))
                    elif not cut.convex_p2:
                        cuts.append((Line(cut.line.p2, -1 * normal), 'flat-p2'))
                    else:
                        # Leave cut in middle
                        cuts.append((Line(cut.line.midpoint + du/2, -1 * normal), 'flat-mid'))
                else:
                    # Will need multiple cuts to fill entire line
                    cut_len = cut.line.direction.length()
                    cuts_required = ceil(cut_len/flat_tool_tip)
                    cut_dir = cut.line.direction.unit()
                    if tool_angle:
                        start_angle = classified[cut_index-1].cut_line.direction.angle()+tool_angle
                        end_angle = classified[(cut_index+1) % len(classified)].cut_line.direction.angle()
                        # TODO-This works, but is there something smarter?
                        delta_angle = (start_angle-end_angle) % 360
                        start_angle = start_angle % 360
                        end_angle = start_angle - delta_angle
                        for t, u in zip(t_range(cuts_required - 1, 0, cut_len - flat_tool_tip),
                                        t_range(cuts_required - 1, start_angle, end_angle)):
                            cut_start = cut.line.p1 + t * cut_dir
                            cut_end = cut_start + cut_dir * flat_tool_tip
                            tool_dir = Vector(cos(radians(u)), sin(radians(u)))
                            cut_end = cut_end - tool_dir * flat_tool_extension
                            # tool_dir = Vector(cos(radians(base_angle)), sin(radians(base_angle)))
                            cuts.append((Line(cut_end, tool_dir * self.module * 2), 'flat-multi-a'))
                    else:
                        # Simple flat cuts work like this:
                        for t in t_range(cuts_required-1, 0, cut_len-tool_tip_height):
                            cut_start = cut.line.p1 + t * cut_dir
                            cut_end = cut_start + cut_dir * tool_tip_height
                            cuts.append((Line(cut_end, -1 * normal), 'flat-multi'))
            else:
                cut_line = cut.cut_line
                if cut.kind == 'flat-tip':
                    # flat-tip acts like descending, so need to reverse the line
                    cut_line = Line(cut.cut_line.p2, cut.cut_line.p1)
                if cut.overshoot:
                    # TODO-this needs to be calculated based on nearby edges
                    allowed_overshoot = 0.2 * self.module
                    overshoot = cut_line.direction.unit() * allowed_overshoot
                    cut_line = Line(cut_line.origin - overshoot, cut_line.direction + overshoot)
                cuts.append((cut_line, cut.kind))

            details = []
            inward = cut.inward() or (cut.kind == 'flat-tip' and tool_angle)
            for adjusted_cut, kind in cuts:
                cut_angle = adjusted_cut.direction.angle()
                if inward:
                    rotation = -tool_angle/2 - cut_angle
                else:
                    rotation = tool_angle/2 - cut_angle

                t = Transform().rotate(-rotation)
                cut_origin = adjusted_cut.origin
                # TODO-This should check for intersections with other edges
                # TODO-replace hacky heuristic to shorten vertical cuts
                if tool_angle and cut.kind in {'in', 'out'}:
                    vertical_cut_reduction = 0
                    # vertical_cut_reduction = 1.5
                    cut_origin += adjusted_cut.direction.unit() * self.module * vertical_cut_reduction
                y, z = t.transform_pt(cut_origin)
                if inward:
                    z = z + half_tool_tip
                else:
                    z = z - half_tool_tip
                details.append(CutDetail(adjusted_cut, kind, rotation, y, z))
            cut.cut_details.extend(details)

        return classified

    def classify_cuts(self, tool_angle, tool_tip_height=0.0) -> List[ClassifiedCut]:
        """
            Classify the cuts in this polygon.
            Calls first pass and then performs error checking and reorients the cut.cut_line to
            tool alignment.

            :param tool_angle: tool angle in degrees
            :param tool_tip_height:
            :returns: List of (cut, cut-kind, convex, allowed overshoot)

            Overshoot is allowed if the cut is along a convex part of the polygon.

            Classify cuts by angle relative to center of gear::

               <--- iterating CCW around top of gear
                              out
                       ending  |    u
                     c         |       n
                    s          |         d
                    a          |          e
               flat ---------- + ---------r-
                    d          |
                    e          |         c
                     s         |       u
                      cending  |    t
                               in

        """
        classified = self.classify_cuts_pass1(tool_angle, tool_tip_height)
        refined = self.classify_cuts_pass2(classified, tool_angle, tool_tip_height)
        return refined

    def cut_params(self, tool_angle, tool_tip_height=0.0) -> List[Tuple[float, float, float, str]]:
        """
            Generate list of (rotation, y-position, z-position) from polygon
            :param tool_angle:          In radians
            :param tool_tip_height:     Tool tip height
            :return: List of (r, y, z, cut-kind)
        """
        classified = self.classify_cuts(tool_angle, tool_tip_height)

        # Rotate classified cuts until we find the first edge after an "in" edge
        orig_len = len(classified)
        last_in = -1
        while not classified[last_in].inward():
            last_in -= 1
        if last_in != -1:
            classified = classified[last_in+1:] + classified[:last_in+1]
        assert len(classified) == orig_len

        cut_params = []
        for outer_cut in classified:
            for detail_cut in outer_cut.cut_details:
                cut_params.append((detail_cut.angle, detail_cut.y, detail_cut.z, detail_cut.kind))
        return cut_params
