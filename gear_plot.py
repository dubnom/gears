import os
from typing import List, Tuple, NamedTuple
import matplotlib.pyplot as plt

from anim.geom import polygon_area, iter_rotate, Line, Vector, Point
from gear_base import PointList, plot
from gear_cycloidal import CycloidalPair
from gear_involute import GearInvolute, InvolutePair

# setenv SHOW_INTERACTIVE to 1 or true to display interactive plots
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

    debug_print = not False
    if debug_print:
        for cut in cuts[:20]:
            print(cut)
    return cuts


class CutError(Exception):
    pass


def plot_classified_cuts(poly: PointList, tool_angle, tool_tip_height=0.0) -> List[Tuple[float, float, float]]:
    classified = classify_cuts(poly, tool_angle, tool_tip_height)

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
                # Will need multiple cuts to fill entire line
                print('TODO: Iterate for multiple cuts: %s' % cut)
                cuts.append((cut.line, 'undercut'))
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

    return []




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


def main():
    #test_cuts(); return
    plot_classified_cuts(CycloidalPair(40, 17).pinion().poly, tool_angle=0.0, tool_tip_height=1/32*25.4); return
    plot_classified_cuts(GearInvolute(11).gen_poly(), 0); return
    do_pinions(zoom_radius=5, cycloidal=not False); return
    # do_gears(zoom_radius=5, pinion_teeth=7, cycloidal=True, animate=True)
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
