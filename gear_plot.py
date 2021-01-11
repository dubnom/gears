from __future__ import annotations
import os
from typing import Optional
import matplotlib.pyplot as plt
from math import cos, sin, radians

from x7.geom.geom import Line, Vector, Point
from x7.geom.utils import arc, plot
from gear_base import GearInstance, CUT_KIND_COLOR_MAP
from gear_cycloidal import CycloidalPair
from gear_involute import GearInvolute, InvolutePair

# setenv SHOW_INTERACTIVE to 1 or true to display interactive plots
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
        for r, y, z, k in gear.cut_params(tool):
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
        from x7.geom.plot_viewer import PlotViewer
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
        from x7.geom.plot_viewer import PlotViewer
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

    pair = InvolutePair(137, 33, module=2, **GearInvolute.HIGH_QUALITY)
    pair.plot()
    pair.wheel().plot_show()
    pair = InvolutePair(137, 33, module=2)
    # pair = InvolutePair(31, 27, module=2, **GearInvolute.HIGH_QUALITY)
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
