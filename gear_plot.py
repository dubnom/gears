from __future__ import annotations
import os
from typing import Tuple, Optional, List, Callable
import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np
from math import cos, sin, tan, tau, pi, radians, hypot, atan2

from anim.geom import Line, Vector, Point, BBox
from gear_base import plot, GearInstance, CUT_KIND_COLOR_MAP
from anim.utils import t_range, arc, circle
from gear_cycloidal import CycloidalPair
from gear_involute import GearInvolute, InvolutePair, Involute

# setenv SHOW_INTERACTIVE to 1 or true to display interactive plots
from rack import Rack
from tool import Tool

SHOW_INTERACTIVE = os.environ.get('SHOW_INTERACTIVE', 'false').lower() in {'1', 'true'}


def plot_classified_cuts(gear: GearInstance, tool: Tool):
    classified = gear.classify_cuts(tool)
    check_cut = True
    if check_cut:
        found = 0
        while not classified[found].inward():
            found += 1
        while not classified[found].outward():
            found += 1
        classified = classified[:found+5]
        check_a = classified[found].cut_line
        while classified[found].kind != 'descending':
            found -= 1
        min_found: Optional[Line] = None
        min_angle = 180.0
        while classified[found].kind != 'ascending':
            check_b = classified[found].cut_line
            check_line = Line(check_a.p1, check_b.p2)
            check_angle = check_a.direction.angle()-check_line.direction.angle()
            if not min_angle or check_angle < min_angle:
                min_found = check_line
                min_angle = check_angle
            found -= 1
        check_len = max(check_a.direction.length(), min_found.direction.length()) * 1.25
        check_a = Line(check_a.origin, check_a.direction.unit()*check_len)
        min_found = Line(min_found.origin, min_found.direction.unit()*check_len)
        plot([check_a.p2, min_found.p1, min_found.p2], 'pink')
        # print(check_a.direction.angle(), min_found.direction.angle())
        min_arc = arc(check_a.direction.length()*0.7, check_a.direction.angle(), min_found.direction.angle(), check_a.p1)
        plot(min_arc, 'pink')
        mid_arc = min_arc[len(min_arc)//2]
        plt.text(mid_arc[0]-0.3, mid_arc[1], '%.1f deg' % min_angle,
                 bbox=dict(facecolor='white', alpha=0.5))
        arc_start = Vector(*classified[0].line.p1.xy()).angle()
        arc_end = Vector(*classified[-1].line.p2.xy()).angle()
        arc_extra = (arc_end - arc_start) * 0.5
        plot(arc(gear.pitch_radius, arc_start-arc_extra, arc_end+arc_extra, Point(0, 0)), 'green')
        plot(arc(gear.tip_radius, arc_start-arc_extra, arc_end+arc_extra, Point(0, 0)), 'yellow')
        if gear.root_radius:
            plot(arc(gear.root_radius, arc_start - arc_extra, arc_end + arc_extra, Point(0, 0)), 'yellow')
        plot(arc(gear.base_radius, arc_start-arc_extra, arc_end+arc_extra, Point(0, 0)), 'brown')
        # plot(circle(gear.pitch_radius+gear.module, Point(0, 0)), 'yellow')
        # plot(circle(gear.pitch_radius-gear.module*1.25, Point(0, 0)), 'yellow')
        # plot(circle(gear.pitch_radius*cos(radians(20)), Point(0, 0)), 'brown')
        plt.title('Check angle for %s' % gear.description())

    for outer_cut in classified:
        # print(outer_cut.kind, outer_cut.cut_line.p1.round(2), outer_cut.cut_line.p2.round(2))
        for detail_cut in outer_cut.cut_details:
            dcd = detail_cut.line.direction
            dcn = dcd.normal().unit()
            if outer_cut.inward():
                dcn = -1 * dcn
            p1 = detail_cut.line.origin
            p2 = p1+dcn*tool.tip_height
            pm = p1.mid(p2)
            # print('  ', detail_cut.kind, detail_cut.line.p1.round(2), detail_cut.line.p2.round(2))
            kind = detail_cut.kind.split('-')[0]
            plot([p1+dcd, p1, pm, pm+dcd, pm, p2, p2+dcd*0.5], CUT_KIND_COLOR_MAP[kind])
    # for outer_cut in classified:
    #     plot([outer_cut.line.p1, outer_cut.line.p2], 'grey')
    plt.axis('equal')
    plt.show()
    print_fake_gcode = False
    if print_fake_gcode:
        for r, y, z, k in gear.cut_params(tool.angle, tool.tip_height):
            print('( kind: %s )' % k)
            print('G_ A%10.4f Y%10.4f Z%10.4f' % (r, y, z))


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


def all_gears(cycloidal=True, animate=True, zoom=False):
    import gear_config

    gears = []
    for planet, (x, y, module) in gear_config.GEARS.items():
        if cycloidal:
            pair = CycloidalPair(x, y, module=module)
        else:
            pair = InvolutePair(x, y, module=module)
        gear_x = pair.wheel()
        gear_y = pair.pinion()
        gears.append((pair, gear_x, gear_y))

    if animate:
        from anim.viewer import PlotViewer
        rotation = [0]

        def update(ax):
            for pair, gear_x, gear_y in gears:
                extra = 0
                if cycloidal:
                    extra = 0.08 if pair.pinion_teeth < 10 else 0.05
                gear_x.set_zoom(plotter=ax)
                if zoom:
                    ax.set_xlim(0, 80)
                    ax.set_ylim(-40, 40)
                rot = rotation[0]/7*gear_x.teeth
                gear_x.plot('blue', rotation=rot, plotter=ax)
                gear_y.plot('green', rotation=-rot + extra, plotter=ax)
            rotation[0] += 0.002
        pv = PlotViewer(update_func=update)
        pv.mainloop()       # never returns
    else:
        for pair, gear_x, gear_y in gears:
            gear_x.plot()
            gear_y.plot(color='green')
        gears[0][1].plot_show()


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
    gear = GearInstance(1, 1, 'Test', 'thingie', poly, Point(0, 0))
    plot_classified_cuts(gear, Tool(angle=0.0, tip_height=0.1))


def pplot(rt, color='black', plotter=None):
    """Polar plot of r, theta"""
    plotter = plotter or plt
    xy = [(r*cos(t), r*sin(t)) for r, t in rt]
    plotter.plot(*zip(*xy), color)


def test_inv(num_teeth=None, do_plot=True):
    def inv(radius=0.0, angle=0.0, offset_angle=0.0, offset_radius=0.0, offset_norm=0.0, clip=None):
        """Involute with offset_radius and offset_norm (which makes this a generalized trochoid)"""
        # x = r * cos(a) + r*(a-oa) * sin(a)
        # x = (r-or) * cos(a) + r*(a-oa-on) * sin(a)
        x = (radius-offset_radius) * cos(angle) + (radius*(angle - offset_angle)-offset_norm) * sin(angle)
        y = (radius-offset_radius) * sin(angle) - (radius*(angle - offset_angle)-offset_norm) * cos(angle)
        # y = self.radius * (sin(angle) - (angle - offset) * cos(angle))
        return (x, y) if clip is None or hypot(x, y) < clip else None

    def pp(t_l: float, t_h: float, fn: Callable, radius=None):
        curve = list(filter(None, [fn(t) for t in t_range(50, t_l, t_h)]))
        if radius:
            if t_l + t_h < 0:
                curve = [(Vector(*curve[0]).unit()*radius).xy()] + curve
            else:
                curve = curve + [(Vector(*curve[-1]).unit() * radius).xy()]
        return curve

    def cross(r, c, color='black'):
        u = Vector(r, r)
        d = Vector(r, -r)
        plot([c - u, c + u, c, c - d, c + d], color)

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
    tip_half_tooth = half_tooth - addendum*tan(radians(pressure_angle))
    # print('tht: ', tip_half_tooth)
    # print('rack.tth: ', rack.tooth_tip_high)
    tr = pitch_radius + addendum
    relief_factor = 1.25
    dr = pitch_radius - addendum * relief_factor
    tall_addendum = addendum * relief_factor
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
        """Trochoid for undercut"""
        offset_angle = 0
        cr = pitch_radius*200
        return inv(angle=t + offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                   offset_radius=addendum, offset_norm=tip_half_tooth, clip=cr)

    def calc_trochoid_end() -> Point:
        # https://math.stackexchange.com/questions/2130212/finding-the-point-of-intersection-of-the-involute-profile-and-trochoidal-root-cu
        # addendum/(delta Y + y_0) == tan(pressure_angle)
        # addendum/(t * pitch_radius + tip_half_tooth) == tan(pressure_angle)
        # addendum/tan(pressure_angle) == t*pitch_radius + tip_half_tooth
        # t*pitch_radius = addendum/tan(pressure_angle) - tip_half_tooth
        # t = (addendum/tan(pressure_angle) - tip_half_tooth)/pitch_radius

        # Calculate trochoid parameter
        t = (addendum/tan(radians(pressure_angle)) - tip_half_tooth)/pitch_radius
        return Point(*f_undercut_edge(-t))

    if not do_plot:
        return

    # tip_half_tooth = half_tooth - addendum * tan(radians(pressure_angle))
    # print('tth: ', tip_half_tooth)
    tl, th = 0, 1
    tl, th = pi/2-0.1, pi/2+0.1
    tl, th = pi/2-0.1, pi/2+0.1
    # for tc in [0, tau, 2*tau]:
    plot(circle(pitch_radius), 'grey')
    plot(circle(base_radius), 'orange')
    plot(circle(tr), 'yellow')
    plot(circle(dr), 'yellow')

    # https://math.stackexchange.com/questions/3791094/how-do-i-find-the-intersection-of-an-involute-gears-involute-face-curves-and-tr
    # gamma_max = -2*(relief_factor-profile_shift) tan(pressure_angle) / num_teeth
    # t_max = gamma_max = -2*relief_factor/num_teeth * tan(pressure_angle)
    def t_max(rf=relief_factor, profile_shift=0):
        return -2 * (rf-profile_shift) / num_teeth * tan(radians(pressure_angle))

    def plot_t_max(t, fna, fnb, color):
        a = fna(-t)
        b = fnb(t)
        cross(0.05, Point(*a), color)
        cross(0.05, Point(*b), color)
        plot([a, b], color)

    def find_intersections(a_vals, b_vals):
        a_low, a_high, a_fn = a_vals
        b_low, b_high, b_fn = b_vals

        def ab_func(ab):
            a, b = ab
            ax, ay = a_fn(a)
            bx, by = b_fn(b)
            return np.array([ax-bx, ay-by])

        result = scipy.optimize.fsolve(ab_func, np.array([(a_low+a_high)/2, (b_low+b_high)/2]), full_output=True)
        from pprint import pp
        pp(result)

        # plot(pp(a_low, a_high, a_fn), 'red')
        # plot(pp(b_low, b_high, b_fn), 'red')

        def offset_points(t_pt, x_offset):
            return [(x + x_offset, y) for t, (x, y) in t_pt]

        for n in range(15):
            steps = 20
            offset = 2 - 1 / (n + 1)
            a_points = [(t, a_fn(t)) for t in t_range(steps, a_low, a_high)]
            a_segs = [(t1, t2, Line(Point(*p1), Point(*p2))) for (t1, p1), (t2, p2) in zip(a_points, a_points[1:])]
            b_points = [(t, b_fn(t)) for t in t_range(steps, b_low, b_high)]
            b_segs = [(t1, t2, Line(Point(*p1), Point(*p2))) for (t1, p1), (t2, p2) in zip(b_points, b_points[1:])]
            plot(offset_points(a_points, offset), 'green')
            plot(offset_points(b_points, offset), 'red')

            # Find an a_segment which intersects a b_segment
            found = False
            for a_t1, a_t2, a_seg in a_segs:
                for b_t1, b_t2, b_seg in b_segs:
                    intersection = a_seg.segment_intersection(b_seg)
                    if intersection:
                        a_low = a_t1
                        a_high = a_t2
                        b_low = b_t1
                        b_high = b_t2
                        found = True
                        break
            threshold = 1e-9        # TODO-should be relative
            if abs(a_high-a_low) < threshold and abs(b_high-b_low) < threshold:
                break
            if not found:
                print('Not Found: n=%d da=%.15f db=%.15f' % (n, a_high - a_low, b_high - b_low))
                break
            # else:
            #     print('Found: da=%.15f db=%.15f' % (a_high - a_low, b_high - b_low))
        # TODO-Use the t_value from the intersection to improve this result
        print('res: %.15f %.15f [%2d iters]  found: %.15f %.15f' %
              (result[0][0], result[0][1], result[1]['nfev'], (a_low + a_high) / 2, (b_low + b_high) / 2))
        # return (a_low + a_high) / 2, (b_low + b_high) / 2
        return result[0][0], result[0][1]

    def under(neg: int, off_r: float, off_n: float, clip=None) -> Callable[[float], Tuple]:
        """Return function for undercut curve"""
        offset_angle = 0
        return lambda t: inv(angle=t + offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                             offset_radius=off_r, offset_norm=neg * off_n, clip=clip)

    oa = 0 - tooth_offset_angle
    th, tl = 0 - 2, 0 + 2
    # TODO-improve initial guesses, probably by using range of expected radius at intersection
    t_u, t_inv = find_intersections(
        (-t_max(), th, under(1, addendum, tip_half_tooth, clip=None)),
        (tl, 0, lambda t: inv(angle=t-oa, radius=base_radius, offset_angle=-oa)))
    cross(0.1, Point(*under(1, addendum, tip_half_tooth, clip=None)(t_u)), 'cyan')

    tt_u, tt_inv = find_intersections(
        (-t_max(relief_factor), th, under(1, tall_addendum, tall_tip_half_tooth, clip=None)),
        (tl, 0, lambda t: inv(angle=t-oa, radius=base_radius, offset_angle=-oa)))

    # print('ht/pr:', half_tooth / pitch_radius, ' hta rad:', half_tooth_angle)
    for tc in [0]:
        th, tl = tc-2, tc+2
        for idx, offset_angle in enumerate(t_range(num_teeth, 0, tau, False)):
            oa = offset_angle - tooth_offset_angle
            plot(pp(-t_inv, th, lambda t: inv(angle=t+oa, radius=base_radius, offset_angle=oa, clip=tr), radius=None), 'darkgreen')
            plot(pp(tl, t_inv, lambda t: inv(angle=t-oa, radius=base_radius, offset_angle=-oa, clip=tr), radius=None), 'darkblue')
            cr = pitch_radius+addendum*0.3

            def under(neg: int, off_r: float, off_n: float, clip=cr) -> Callable[[float], Tuple]:
                """Return function for undercut curve"""
                return lambda t: inv(angle=t+offset_angle, radius=pitch_radius, offset_angle=offset_angle,
                                     offset_radius=off_r, offset_norm=neg*off_n, clip=clip)

            tm = t_max(1.25)
            plot(pp(-tm, tt_u, under(1, tall_addendum, tall_tip_half_tooth)), 'lightblue')
            plot(pp(-tt_u, tm, under(-1, tall_addendum, tall_tip_half_tooth)), 'lightgreen')
            plot_t_max(tm, under(1, tall_addendum, tall_tip_half_tooth), under(-1, tall_addendum, tall_tip_half_tooth), 'lightgreen')

            tm = t_max()
            plot_t_max(tm, under(1, addendum, tip_half_tooth), under(-1, addendum, tip_half_tooth), 'green')
            # Make the curve a bit longer for plotting
            tmu = t_u+tm
            t_u, tm = t_u+tmu, tm+tmu
            plot(pp(-tm, t_u, under(1, addendum, tip_half_tooth)), 'blue')
            plot(pp(-t_u, tm, under(-1, addendum, tip_half_tooth)), 'green')

    # pplot(pp(-1, 1, lambda t: (pitch_radius * (1 - t * tan(radians(pressure_angle))), t)), 'pink')
    gi = GearInvolute(teeth=num_teeth, module=module, pressure_angle=pressure_angle)
    # gi.plot('red', gear_space=True)
    # plot(pp(-1, 1, lambda t: (pitch_radius - t * tan(radians(pressure_angle)), t)), 'red')
    plt.axis('equal')
    plt.title('Involutes & Throchoids for teeth=%d' % num_teeth)
    plt.show()


def find_nt():
    module = 1
    for nt in range(5, 45):
        pr = module * nt / 2
        dr = pr - 1.25 * module
        br = cos(radians(20)) * pr
        print('%-3d %8.4f %8.4f %8.4f %s' % (nt, pr, dr, br, br < dr))


def main():
    # find_nt(); return
    # [test_inv(n) for n in [3, 7, 17, 21, 101]]; return
    # [test_inv(n) for n in range(3, 18, 3)]; return
    # test_inv(137); test_inv(42); test_inv(142); return
    # test_cuts(); return
    # all_gears(cycloidal=not False); return
    # do_gears(zoom_radius=5, wheel_teeth=137, pinion_teeth=5, cycloidal=True, animate=True); return

    # pair = InvolutePair(137, 33, module=2)
    pair = InvolutePair(31, 27, module=2)
    # pair = CycloidalPair(137, 33, module=0.89647)
    tool = Tool(angle=0.0, tip_height=1/32*25.4)
    plot_classified_cuts(pair.wheel(), tool)
    plot_classified_cuts(pair.pinion(), tool)
    pair.plot()
    pair.wheel().plot_show()
    return
    plot_classified_cuts(CycloidalPair(40, 17).pinion(), tool_angle=0.0, tool_tip_height=1/32*25.4); return
    # plot_classified_cuts(GearInvolute(11).gen_poly(), 0); return
    # do_pinions(zoom_radius=5, cycloidal=not False); return
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
