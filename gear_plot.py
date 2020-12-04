import os
from typing import List, Tuple, NamedTuple
import matplotlib.pyplot as plt
import scipy.optimize
from math import cos, sin, tan, tau, pi, radians, hypot, atan2, sqrt, degrees, ceil

from anim.geom import polygon_area, iter_rotate, Line, Vector, Point
from anim.transform import Transform
from gear_base import PointList, plot, t_range, circle, arc
from gear_cycloidal import CycloidalPair
from gear_involute import GearInvolute, InvolutePair, Involute

# setenv SHOW_INTERACTIVE to 1 or true to display interactive plots
from rack import Rack

SHOW_INTERACTIVE = os.environ.get('SHOW_INTERACTIVE', 'false').lower() in {'1', 'true'}
COLOR_MAP = {
    'undercut': 'red',
    'ascending': '#0000FF',
    'descending': '#00FF00',
    'out': '#9999FF',
    'in': '#99FF99',
    'flat': 'orange'
}


class ClassifiedCut:
    """
        A cut classified by direction
    """
    def __init__(self, line: Line, kind: str, convex_p1: bool, convex_p2: bool, overshoot: float, z_offset: float):
        """
            :param line:        Line segment in CCW direction
            :param kind:        in/out/ascending/descending/flat/undercut
            :param convex_p1:      True if shape is convex at line.p1
            :param convex_p2:      True if shape is convex at line.p2
            :param overshoot:   Allowed overshoot
            :param z_offset:    Z offset for tool
        """
        self.line = line
        self.kind = kind
        self.convex_p1 = convex_p1
        self.convex_p2 = convex_p2
        self.overshoot = overshoot
        self.z_offset = z_offset
        # cut_line is reversed if this is an inward cut
        self.cut_line = Line(line.p2, line.p1) if self.inward() else self.line

    def __str__(self):
        c1 = 'convex1' if self.convex_p1 else 'concave1'
        c2 = 'convex2' if self.convex_p2 else 'concave2'
        return '%-10s %-8s %-8s %8.3f %8.3f %s' % (self.kind, c1, c2, self.overshoot, self.radius(), self.cut_line)

    def radius(self):
        """Radius of the deepest point of this cut"""
        return Vector(*self.cut_line.origin.xy()).length()

    def flat(self):
        """True if kind is 'flat'"""
        return self.kind == 'flat'

    def inward(self):
        """True if kind is 'in' or 'descending'"""
        return self.kind in {'in', 'descending'}

    def outward(self):
        """True if kind is 'out' or 'ascending'"""
        return self.kind in {'out', 'ascending'}


def classify_cuts(poly: PointList, tool_angle, tool_tip_height=0.0) -> List[ClassifiedCut]:
    """
        Classify the cuts in this polygon.

        :param poly: List of Points in CCW order
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
    # Is polygon in CCW order?
    if polygon_area(poly) < 0:
        poly = list(reversed(poly))
    half_tool_tip = tool_tip_height / 2
    assert tool_angle == 0.0

    cuts = []

    # These values should depend on tool angle
    angle_eps = 1.0     # degrees
    flat_eps = 15.0     # degrees
    for a, b, c in iter_rotate(poly, 3):
        cut = Line(a, b)
        radial_angle = Vector(*a.mid(b).xy()).angle()
        line_angle = cut.direction.angle()
        delta = line_angle - radial_angle
        if delta > 180:
            delta -= 360
        if delta < -180:
            delta += 360
        # print('a:%-8s b:%-8s ra:%5.1f la:%5.1f d:%5.1f' % (a, b, radial_angle, line_angle, delta))
        if abs(delta) < angle_eps:
            kind = 'out'
        elif 180 - abs(delta) < angle_eps:
            kind = 'in'
        elif abs(delta-90) < flat_eps:
            kind = 'flat'
        elif 0 < delta < 90:
            kind = 'ascending'
        elif 90 < delta < 180:
            kind = 'descending'
        else:
            kind = 'undercut'
        convex_p1 = cuts[-1].convex_p2 if cuts else False
        convex_p2 = cut.direction.cross(c-b) >= 0
        # TODO-overshoot should look for intersections with other segments
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


class CutError(Exception):
    pass


def plot_classified_cuts(poly: PointList, tool_angle, tool_tip_height=0.0):
    classified = classify_cuts(poly, tool_angle, tool_tip_height)
    check_cut = not True
    if check_cut:
        # classified = classified[:16]      # for pinion
        classified = classified[:17]        # for wheel
        check_a = classified[-2].cut_line
        check_b = classified[-7].cut_line
        check_line = Line(check_a.p1, check_b.p2)
        check_angle = check_a.direction.angle()-check_line.direction.angle()
        plot([check_a.p2, check_line.p1, check_line.p2], 'pink')
        print(check_a.direction.angle(), check_line.direction.angle())
        plot(arc(check_a.direction.length()*0.7, check_a.direction.angle(), check_line.direction.angle(), check_a.p1), 'pink')
        plt.text(*(check_line.p1+Vector(1.2, 0.0)).xy(), '%.1f deg' % check_angle,
                 bbox=dict(facecolor='white', alpha=0.5))

    for cut in classified:
        normal = cut.cut_line.direction.unit().normal() * cut.z_offset
        if cut.flat():
            cuts = []
            length = cut.line.direction.length()
            if length == 0:
                continue
            elif cut.line.direction.length() < tool_tip_height:
                # Cut shorter than saw kerf
                if not cut.convex_p1 and not cut.convex_p2:
                    print('ERROR: Cut is too narrow for saw width: %s / saw: %.4f' % (cut, tool_tip_height))
                    print('   length: %.8f' % length)
                    cuts.append((cut.line, 'undercut'))
                if not cut.convex_p1:
                    # Align cut to p1
                    cuts.append((Line(cut.line.p1, cut.line.direction.unit() * tool_tip_height), 'flat'))
                elif not cut.convex_p2:
                    cuts.append((Line(cut.line.p2, -cut.line.direction.unit() * tool_tip_height), 'flat'))
                else:
                    # Leave cut in middle
                    cuts.append((cut.line, 'flat'))
            else:
                cut_len = cut.line.direction.length()
                cuts_required = ceil(cut_len/tool_tip_height)
                cut_dir = cut.line.direction.unit()
                for t in t_range(cuts_required-1, 0, cut_len-tool_tip_height):
                    cut_start = cut.line.p1 + t * cut_dir
                    cut_end = cut_start + cut_dir * tool_tip_height
                    cuts.append((Line(cut_start, cut_end), 'flat'))
            for cut_line, kind in cuts:
                du = cut_line.direction.unit() * tool_tip_height / 2
                mid = cut_line.midpoint
                mid1 = mid - du
                mid2 = mid + du
                plot([cut_line.p1 + normal/2, cut_line.p1,
                      mid1, mid1-normal*2, mid1,
                      mid, mid-normal, mid,
                      mid2, mid2-normal*2, mid2,
                      cut_line.p2, cut_line.p2 + normal/2],
                     COLOR_MAP[kind])
        else:
            pm1 = cut.cut_line.p1 + normal
            pm2 = cut.cut_line.p2 + normal
            plot([pm2, pm1, cut.cut_line.p1, cut.cut_line.p2], COLOR_MAP[cut.kind])
    plt.axis('equal')
    plt.show()
    print_fake_gcode = False
    if print_fake_gcode:
        for r, y, z in cut_params_from_polygon(poly, tool_angle, tool_tip_height):
            print('G_ A%10.4f Y%10.4f Z%10.4f' % (r, y, z))


def cut_params_from_polygon(poly: PointList, tool_angle, tool_tip_height=0.0) -> List[Tuple[float, float, float]]:
    """
        Generate list of (rotation, y-position, z-position) from polygon
        :param poly:                Polygon to cut
        :param tool_angle:          In radians
        :param tool_tip_height:     Tool tip height
        :return: List of (r, y, z)
    """
    assert tool_angle == 0
    half_tool_tip = tool_tip_height / 2
    classified = classify_cuts(poly, tool_angle, tool_tip_height)

    # Rotate classified cuts until we find a the first edge after an "in" edge
    orig_len = len(classified)
    last_in = -1
    while classified[last_in].kind != "in":
        last_in -= 1
    if last_in != -1:
        classified = classified[last_in+1:] + classified[:last_in+1]
    assert len(classified) == orig_len

    cut_params = []
    for cut in classified:
        cuts = []
        if cut.kind == 'undercut':
            raise ValueError("Can't do undercuts yet")
        elif cut.flat():
            normal = cut.cut_line.direction.unit().normal() * cut.z_offset
            du = cut.line.direction.unit() * tool_tip_height
            length = cut.line.direction.length()
            if length == 0:
                continue
            elif cut.line.direction.length() < tool_tip_height:
                # Cut shorter than saw kerf
                if not cut.convex_p1 and not cut.convex_p2:
                    print('ERROR: Cut is too narrow for saw width: %s / saw: %.4f' % (cut, tool_tip_height))
                    print('   length: %.8f' % length)
                    raise ValueError('ERROR: Cut is too narrow for saw width: %s / saw: %.4f' % (cut, tool_tip_height))
                if not cut.convex_p1:
                    # Align cut to p1
                    cuts.append((Line(cut.line.p1 + du, -1 * normal), 'flat'))
                elif not cut.convex_p2:
                    cuts.append((Line(cut.line.p2, -1 * normal), 'flat'))
                else:
                    # Leave cut in middle
                    cuts.append((Line(cut.line.midpoint + du/2, -1 * normal), 'flat'))
            else:
                # Will need multiple cuts to fill entire line
                cut_len = cut.line.direction.length()
                cuts_required = ceil(cut_len/tool_tip_height)
                cut_dir = cut.line.direction.unit()
                for t in t_range(cuts_required-1, 0, cut_len-tool_tip_height):
                    cut_start = cut.line.p1 + t * cut_dir
                    cut_end = cut_start + cut_dir * tool_tip_height
                    cuts.append((Line(cut_end, -1 * normal), 'flat'))

        else:
            cuts.append((cut.cut_line, cut.kind))

        for c, kind in cuts:
            cut_angle = c.direction.angle()
            rotation = tool_angle - cut_angle
            # print('ca=%9.6f rot=%9.6f' % (cut_angle, rotation))
            t = Transform().rotate(-rotation)
            y, z = t.transform_pt(c.origin)
            if kind in {'in', 'descending'}:
                z = z + half_tool_tip
            else:
                z = z - half_tool_tip
            cut_params.append((rotation, y, z))

    return cut_params


def do_pinions(zoom_radius=0., cycloidal=True):
    wheel = None
    color_index = 0
    colors = ['blue', 'red', 'green', 'orange']
    for pt in [6, 8, 10, 12]:  # range(5, 11):
        t1 = 40
        t2 = pt
        if cycloidal:
            pair = CycloidalPair(t1, t2, module=1.0)
        else:
            pair = InvolutePair(t1, t2, module=1.0)
        wheel = pair.wheel()
        pinion = pair.pinion()

        color = colors[color_index % len(colors)]
        # wheel.plot(color)
        pinion.plot(color, rotation=0.5)
        color_index += 1
    wheel.plot_show(zoom_radius)


def do_gears(rot=0., zoom_radius=0., cycloidal=True, wheel_teeth=40, pinion_teeth=6, animate=False):
    t1 = wheel_teeth
    t2 = pinion_teeth
    # GearInvolute(t1, rot=rot, module=1).plot(mill_space=True)
    if cycloidal:
        pair = CycloidalPair(t1, t2, module=1.0)
        wheel = pair.wheel()
        pinion = pair.pinion()
    else:
        pair = InvolutePair(t1, t2, module=1.0)
        wheel = pair.wheel()
        pinion = pair.pinion()

    if animate:
        from anim.viewer import PlotViewer
        rotation = [0]
        if cycloidal:
            extra = 0.08 if pinion.teeth < 10 else 0.05
        else:
            extra = 0.5 if pinion.teeth % 2 == 0 else 0.0

        def update(ax):
            wheel.set_zoom(zoom_radius=zoom_radius, plotter=ax)
            wheel.plot('blue', rotation=rotation[0], plotter=ax)
            pinion.plot('green', rotation=-rotation[0] + extra, plotter=ax)
            rotation[0] += 0.01
        pv = PlotViewer(update_func=update)
        pv.mainloop()       # never returns

    else:
        wheel.plot('blue', rotation=rot)
        if pinion:
            pinion.plot('green')
        wheel.plot_show(zoom_radius)


def all_gears(zoom_radius=0., cycloidal=True, animate=False):
    import gear_config
    assert cycloidal

    for planet, (x, y, module) in gear_config.GEARS.items():
        if x > y:
            pair = CycloidalPair(x, y, module=module)
            gear_x = pair.wheel()
            gear_y = pair.pinion()
        elif x == y:
            pair = CycloidalPair(x, y, module=module)
            gear_x = pair.wheel()
            gear_y = pair.wheel()
            gear_y.center = pair.pinion().center
            gear_y = pair.pinion()
        else:
            pair = CycloidalPair(y, x, module=module)
            gear_x = pair.pinion()
            gear_y = pair.wheel()

        gear_x.plot()
        gear_y.plot(color='green')
    gear_x.plot_show()

    if False and animate:
        from anim.viewer import PlotViewer
        rotation = [0]
        if cycloidal:
            extra = 0.08 if pinion.teeth < 10 else 0.05
        else:
            extra = 0.5 if pinion.teeth % 2 == 0 else 0.0

        def update(ax):
            wheel.set_zoom(zoom_radius=zoom_radius, plotter=ax)
            wheel.plot('blue', rotation=rotation[0], plotter=ax)
            pinion.plot('green', rotation=-rotation[0] + extra, plotter=ax)
            rotation[0] += 0.01
        pv = PlotViewer(update_func=update)
        pv.mainloop()       # never returns

    else:
        wheel.plot('blue', rotation=rot)
        if pinion:
            pinion.plot('green')
        wheel.plot_show(zoom_radius)


def test_cuts():
    points = [
        (-1, -1),
        (-1, 1),
        (0, .5),
        (0, 1.5),
        (1, 1),
        (1, -1),
    ]
    poly = [Point(*xy) for xy in points]
    plot_classified_cuts(poly, 0)


def pplot(rt, color='black', plotter=None):
    """Polar plot of r, theta"""
    plotter = plotter or plt
    xy = [(r*cos(t), r*sin(t)) for r, t in rt]
    plotter.plot(*zip(*xy), color)


def test_inv(num_teeth=None):
    def inv(radius=0.0, angle=0.0, offset_angle=0.0, offset_radius=0.0, offset_norm=0.0, clip=None):
        # x = r * cos(a) + r*(a-oa) * sin(a)
        # x = (r-or) * cos(a) + r*(a-oa-on) * sin(a)
        x = (radius-offset_radius) * cos(angle) + (radius*(angle - offset_angle)-offset_norm) * sin(angle)
        y = (radius-offset_radius) * sin(angle) - (radius*(angle - offset_angle)-offset_norm) * cos(angle)
        # y = self.radius * (sin(angle) - (angle - offset) * cos(angle))
        return (x, y) if clip is None or hypot(x, y) < clip else None

    def pp(t_l, t_h, fn, radius=None):
        curve = list(filter(None, [fn(t) for t in t_range(50, t_l, t_h)]))
        if radius:
            if t_l + t_h < 0:
                curve = [(Vector(*curve[0]).unit()*radius).xy()] + curve
            else:
                curve = curve + [(Vector(*curve[-1]).unit() * radius).xy()]
        return curve

    module = 1
    num_teeth = num_teeth or 17

    tooth_angle = tau / num_teeth / 2
    half_tooth_angle = tooth_angle / 2
    pitch_radius = module * num_teeth / 2
    pitch = module * pi
    pressure_angle = 20
    base_radius = pitch_radius*cos(radians(pressure_angle))
    addendum = module
    half_tooth = pitch / 4
    rack = Rack(module=module, pressure_angle=pressure_angle)
    tip_half_tooth = half_tooth - addendum*tan(radians(pressure_angle))
    # print('tht: ', tip_half_tooth)
    # print('rack.tth: ', rack.tooth_tip_high)
    tr = pitch_radius + addendum
    dr = pitch_radius - addendum * 1.25
    tall_addendum = addendum * 1.25
    tall_tip_half_tooth = half_tooth - tall_addendum * tan(radians(pressure_angle))

    # Calc pitch point where involute intersects pitch circle and offset
    involute = Involute(base_radius, tr, dr)
    pp_inv_angle = involute.calc_angle(pitch_radius)
    ppx, ppy = involute.calc_point(pp_inv_angle)
    pp_off_angle = atan2(ppy, ppx)
    # Multiply pp_off_angle by pr to move from angular to pitch space
    tooth_offset_angle = half_tooth_angle - pp_off_angle

    def f_tooth_edge(theta):
        offset_angle = 0
        oa = offset_angle - tooth_offset_angle
        tr = pitch_radius*20
        return inv(angle=theta - oa, radius=base_radius, offset_angle=-oa, clip=tr)

    def f_undercut_edge(t):
        offset_angle = 0
        cr = pitch_radius*200
        return inv(angle=t + offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                                      offset_radius=addendum, offset_norm=tip_half_tooth, clip=cr)

    def solve_this_radius(t):
        x, y = f_undercut_edge(t)
        x2, y2 = f_tooth_edge(atan2(y, x))
        return hypot(x, y) - hypot(x2, y2)

    def solve_this_distance(t):
        x, y = f_undercut_edge(t)
        x2, y2 = f_tooth_edge(atan2(y, x))
        return hypot(x-x2, y-y2)

    def solve_this(fn, low, high, epsilon=1e-8):
        found: scipy.optimize.RootResults
        found = scipy.optimize.root_scalar(fn, bracket=(low, high), method='brentq', xtol=epsilon)
        # print(found)
        if not found.converged:
            print('Not-solved')
        return found.root

    solution = solve_this(solve_this_radius, -2, 0)
    better_solution: scipy.optimize.OptimizeResult
    better_solution = scipy.optimize.minimize_scalar(
        solve_this_distance, bounds=(solution-0.3, solution+0.3), tol=1e-8)
    print('Solved [%3d]: %9.4f %9.4f %9.4f %9.4f' % (
        num_teeth, solution, better_solution.x,
        solve_this_distance(solution), solve_this_distance(better_solution.x)))
    if num_teeth != 7:
        return
    print(better_solution)

    plot([(solution, -1), (solution, 1)])
    curve = [(t, solve_this_distance(t)) for t in t_range(50, solution-1, solution+1)]
    plot(curve, 'green')
    curve = [(t, solve_this_radius(t)) for t in t_range(50, solution-1, solution+1)]
    plot(curve, 'blue')
    plt.show()

    # tip_half_tooth = half_tooth - addendum * tan(radians(pressure_angle))
    #print('tth: ', tip_half_tooth)
    tl, th = 0, 1
    tl, th = pi/2-0.1, pi/2+0.1
    tl, th = pi/2-0.1, pi/2+0.1
    # for tc in [0, tau, 2*tau]:
    plot(circle(pitch_radius), 'grey')
    plot(circle(base_radius), 'orange')
    plot(circle(tr), 'yellow')
    plot(circle(dr), 'yellow')
    undercut_point = Point(*f_undercut_edge(solution))
    print('Undercut at ', undercut_point)
    plot(circle(0.1, undercut_point), 'pink')

    def cross(r, c, color='black'):
        u = Vector(r, r)
        d = Vector(r, -r)
        plot([c-u, c+u, c, c-d, c+d], color)
    cross(0.05, undercut_point, 'pink')
    # print('ht/pr:', half_tooth / pitch_radius, ' hta rad:', half_tooth_angle)
    for tc in [0]:
        th, tl = tc-1, tc+1
        th, tl = tc-2, tc+2
        # th, tl = tc-6, tc+6
        for offset_angle in t_range(num_teeth, 0, tau, False):
            oa = offset_angle - tooth_offset_angle
            plot(pp(0, th, lambda t: inv(angle=t+oa, radius=base_radius, offset_angle=oa, clip=tr), radius=dr), 'darkgreen')
            plot(pp(tl, 0, lambda t: inv(angle=t-oa, radius=base_radius, offset_angle=-oa, clip=tr), radius=dr), 'darkblue')
            # plot(pp(th, tl, lambda t: inv(angle=t+offset_angle, radius=base_radius, offset_angle=offset_angle)), 'black')
            cr = pitch_radius+addendum*0.3

            plot(pp(tl/4, th, lambda t: inv(angle=t+offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                                            offset_radius=tall_addendum, offset_norm=tall_tip_half_tooth, clip=cr)), 'lightblue')
            plot(pp(tl, th/4, lambda t: inv(angle=t-offset_angle, radius=pitch_radius, offset_angle=-offset_angle,
                                            offset_radius=tall_addendum, offset_norm=-tall_tip_half_tooth, clip=cr)), 'lightgreen')
            plot(pp(tl/4, th, lambda t: inv(angle=t+offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                                            offset_radius=addendum, offset_norm=tip_half_tooth, clip=cr)), 'blue')
            plot(pp(tl, th/4, lambda t: inv(angle=t-offset_angle, radius=pitch_radius, offset_angle=-offset_angle,
                                            offset_radius=addendum, offset_norm=-tip_half_tooth, clip=cr)), 'green')
    # pplot(pp(-1, 1, lambda t: (pitch_radius * (1 - t * tan(radians(pressure_angle))), t)), 'pink')
    gi = GearInvolute(teeth=num_teeth, module=module, pressure_angle=pressure_angle)
    # gi.plot('red', gear_space=True)
    # plot(pp(-1, 1, lambda t: (pitch_radius - t * tan(radians(pressure_angle)), t)), 'red')
    plt.axis('equal')
    plt.show()


def main():
    # [test_inv(n) for n in range(3, 34)]; return
    # test_inv(); return
    # test_cuts(); return
    all_gears(); return
    do_gears(zoom_radius=5, wheel_teeth=137, pinion_teeth=5, cycloidal=True, animate=True); return

    cp = CycloidalPair(137, 33)
    plot_classified_cuts(cp.wheel().poly, tool_angle=0.0, tool_tip_height=1/32*25.4)
    plot_classified_cuts(cp.pinion().poly, tool_angle=0.0, tool_tip_height=1/32*25.4)
    return
    plot_classified_cuts(CycloidalPair(40, 17).pinion().poly, tool_angle=0.0, tool_tip_height=1/32*25.4); return
    #plot_classified_cuts(GearInvolute(11).gen_poly(), 0); return
    #do_pinions(zoom_radius=5, cycloidal=not False); return
    do_gears(zoom_radius=5, pinion_teeth=7, cycloidal=True, animate=True)
    do_gears(zoom_radius=7, pinion_teeth=18, animate=True, cycloidal=False); return
    for pt in [5, 6, 7, 8, 9, 10, 11]:
        do_gears(zoom_radius=5, pinion_teeth=pt)
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
