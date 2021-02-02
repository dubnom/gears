"""
Various plots to document gear calculations
See also: https://www.tec-science.com/mechanical-power-transmission/involute-gear/calculation-of-involute-gears/
"""
from math import cos, sin, radians
from typing import Union, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from x7.geom.geom import PointList, Point, Vector, XYList, XYTuple
from x7.geom.utils import plot, circle, plus, cross, plot_show, min_rotation
from x7.lib.iters import t_range

from gear_involute import GearInvolute, Involute, InvoluteWithOffsets


def path_close(path: PointList) -> PointList:
    return path + [path[0]]


def arc(r, sa, ea, c=Point(0, 0), steps=-1, direction='shortest',
        arrow=None, mid=False) -> Union[XYList, Tuple[XYList, XYTuple]]:
    """
        Generate an arc of radius r about c as a list of x,y pairs
        :param r:   radius
        :param sa:  starting angle in degrees
        :param ea:  ending angle in degrees
        :param c:   center point of arc
        :param steps: number of steps (-1 => 1 per degree)
        :param direction: 'shortest', 'ccw' (from sa to ea), 'cw' (from sa to ea) (None defaults to 'shortest')
        :param arrow: None->no arrow, 'both', 'start', 'end'
        :param mid: Also return arc midpoint
        :return: list of (x, y) or (list of (x,y), arc-mid-vector) if midpoint
    """
    sa = sa % 360
    ea = ea % 360
    if direction == 'ccw':
        if ea < sa:
            ea += 360
    elif direction == 'cw':
        if ea > sa:
            ea -= 360
    elif direction == 'shortest' or direction is None:
        ea = sa + min_rotation(ea, sa)
    else:
        raise ValueError('direction: expected shortest, ccw, or cw, not %r' % direction)
    steps = int(abs(ea-sa)+1) if steps < 0 else steps
    path = [(r * cos(t) + c.x, r * sin(t) + c.y) for t in t_range(steps, radians(sa), radians(ea))]
    if arrow in ('both', 'end'):
        arc_len = abs(radians(ea-sa)) * r
        arrow_len = min(arc_len / 3, r / 4)
        last = Point(*path[-1])
        v = Point(*path[-5]) - last
        v = v.unit() * arrow_len
        vl = v.rotate(15)
        vr = v.rotate(-15)
        path.extend([last+vl, last, last+vr])
    elif arrow in ('both', 'start'):
        pass
    else:
        raise ValueError('arrow: must be None, start, end, or both, not %r' % arrow)
    if mid:
        t = radians(sa+ea) / 2
        mid = (r * cos(t), r * sin(t))
        return path, mid
    else:
        return path


def angle(text: str, center: Point, p1: Union[Vector, Point],
          p2: Union[Vector, Point, None] = None, angle: Optional[float] = None,
          xytext: Optional[XYTuple] = None, vlen: Optional[float] = None, arc_pos=None,
          color='cyan', direction=None,
          arc_color=None, vec_color=None,
          ):
    r"""
        Annotate an angle.  ``text`` may specify how angle (in degrees) is to be formatted.
        ``@`` will be substituted with ``${angle:.1f}^\circ$`` (use ``{at}`` to get a single ``@`` sign).
        Examples::

        *   ``r'$\alpha is {angle:.1f}^\circ$'``
        *   ``r'$\alpha is $@'``
        *   ``'angle is {angle:8.3f} degrees'``

        :param text:    Text to display, with optional degrees spec.
        :param center:  Center point
        :param p1:      First point or vector to first point
        :param p2:      Second point or vector (exclusive with angle)
        :param angle:   Angle to second point (exclusive with p2)
        :param xytext:  Override default xytext location
        :param vlen:    Override displayed p1 & p2 vector lengths
        :param arc_pos: Position of angular arc along vectors (default 0.5)
        :param color:   Color of arc & vectors
        :param arc_color: Color of arc (defaults to color, 'none' for no arc)
        :param vec_color: Color of vectors (defaults to color, 'none' for no vectors)
        :param direction: Direction of arc (ccw, cw, shortest).  None defaults to shortest
        :return: value of plt.annotate()
    """
    """Annotate an angle"""
    p1v = p1 if isinstance(p1, Vector) else (p1 - center)
    if p2 is None:
        if angle is None:
            # angle is required
            raise ValueError('One of p2 or angle must be supplied')
        p2v = p1v.rotate(angle)
        if direction is None:
            direction = 'ccw' if angle > 0 else 'cw'
    else:
        if angle is not None:
            raise ValueError('Only one of p2 or angle can be supplied')
        p2v = p2 if isinstance(p2, Vector) else (p2 - center)
    arc_color = arc_color or color
    vec_color = vec_color or color
    vlen = vlen if vlen else max(p1v.length(), p2v.length())
    if vec_color != 'none':
        plot([center+p1v.unit()*vlen, center, center+p2v.unit()*vlen], color=vec_color)
    if angle is None:
        angle = (p2v.angle() - p1v.angle()) % 360
    text = text.replace('@', r'${angle:.1f}^\circ$')
    text = text.format(angle=angle, at='@')
    angle = abs(angle)

    if arc_pos is None:
        if angle < 15:
            arc_pos = 0.95
        elif angle < 40:
            arc_pos = 0.75
        else:
            arc_pos = 0.5
    arc_path, midv = arc(arc_pos*vlen, p1v.angle(), p2v.angle(), center, direction=direction, arrow='both', mid=True)
    if arc_color != 'none':
        plot(arc_path, color=arc_color)
    midv = Vector(*midv)
    # plot([center, center + midv])
    xo = -1 if midv.x < 0 else 1
    yo = -1 if midv.y < 0 else 1
    xytext = (10*xo, 20*yo) if xytext is None else xytext

    return plt.annotate(
        text, (center+midv).xy(),
        xytext=xytext, textcoords='offset points',
        ha='left' if xytext[0] >= 0 else 'right',
        va='bottom' if xytext[1] >= 0 else 'top',
        arrowprops=dict(arrowstyle='->'),
        bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha=0.8),
    )


def doc_radii():
    def legend2():
        line, = plot([(0, 0)], color='white')
        leg1 = plt.legend(loc=3)
        leg2: Legend = plt.legend([line], [extra], loc=2)
        leg2.set_title('Inputs')
        plt.gca().add_artist(leg1)

    rot = -0.35

    g = GearInvolute(17, profile_shift=0.5, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot(g.gen_rack_tooth(teeth=5, rot=g.rot+0.5), color='lightgrey')
    extra = g.plot(gear_space=False, mill_space=False, pressure_line=True, linewidth=1)
    plt.title('Involute Radii with Profile Shift of 0.5')
    legend2()
    g.plot_show((6, 1, 3))

    g = GearInvolute(17, profile_shift=0, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot(g.gen_rack_tooth(teeth=5, rot=g.rot+0.5), color='lightgrey')
    extra = g.plot(gear_space=False, mill_space=False, pressure_line=False, linewidth=1)
    plt.title('Involute Radii')
    legend2()
    g.plot_show((5.5, 1.5, 3))


def doc_tooth_parts():
    for teeth in [7, 27]:
        g = GearInvolute(teeth, **GearInvolute.HIGH_QUALITY)
        g.curved_root = False
        plot(g.instance().poly, 'lightgrey')
        plt.title('Tooth path for %d teeth' % g.teeth)
        plot(g.gen_rack_tooth(teeth=5, rot=0.5), color='c', linestyle=':')
        colors = dict(face='blue', root_cut='red', dropcut='orange', root_arc='green', tip_arc='cyan')
        labels = dict(face='Gear Face', root_cut='Root Cut', dropcut='Drop Cut', root_arc='Root Arc', tip_arc='Tip Arc')
        seen = set()
        parts = g.gen_gear_tooth_parts(closed=True, include_extras=True)
        for tag, points in parts:
            if tag.startswith('_'):
                continue
            plot(points, color=colors[tag], linewidth=3, label=labels[tag] if tag not in seen else None)
            seen.add(tag)
            if tag == 'root_arc':
                # Plot this again mirrored so that plot is symmetric
                plot([(p.x, -p.y) for p in points], color=colors[tag], linewidth=3)
        plt.legend(loc=6)
        g.plot_show((3, 1, 2))


def test_angle():
    # plot(arc(5, 45, 45+270, Point(2, 2)), color='cyan')
    angle('Something is @', Point(-20, 0), Vector(10, 1), Vector(8, 3), color='green', arc_pos=0.95)
    angle(r'$\alpha$ is @', Point(-20, 0), Vector(8, 4), Vector(-8, 4), color='green', arc_pos=0.95)
    angle('Hi-precision is {angle:8.5f}', Point(-20, 0), Vector(8, 4), Vector(-8, 4), color='green', arc_pos=0.5)
    angle('Something Blue', Point(0, 0), Point(1, 0), Point(0, 1), vlen=15, color='blue')
    angle('Something Else', Point(20, 0), Vector(10, 1), angle=-270)
    angle('15,0 < 0,15', Point(0, 20), Vector(5, 0), Vector(0, 5), color='red', direction='ccw')
    angle('0,15 < 15,0', Point(0, -20), Vector(0, 5), Vector(5, 0), color='pink', direction='ccw')
    angle('ang 10: @', Point(-30, -20), Vector(10, 10), angle=10, color='pink', arc_color='red', direction='ccw')
    angle('ang 30{at}@', Point(-30, -20), Vector(10, 10), angle=30, color='pink', vec_color='red', xytext=(1, -40), direction='ccw')
    angle('ang 100: @', Point(-30, -20), Vector(10, 10), angle=100, color='pink', direction='ccw')
    angle('Just Arc', Point(20, -20), Vector(10, 10), angle=90, color='pink', vec_color='none', direction='ccw')
    angle('Just Vecs', Point(20, -20), Vector(-10, -10), angle=90, color='pink', arc_color='none', direction='ccw')
    plot([(0, -30)], color='white')
    plt.axis('equal')
    plt.grid()
    plt.show()


def doc_tooth_equations():
    for teeth in [7, 127]:
        g = GearInvolute(teeth, **GearInvolute.HIGH_QUALITY)
        plot(path_close(g.instance().poly), 'lightgrey')
        plt.title('Tooth path for %d teeth' % g.teeth)
        plot(g.gen_rack_tooth(teeth=5, rot=0.5), color='c', linestyle=':')
        colors = dict(face='blue', root_cut='red', dropcut='orange', root_arc='green', tip_arc='cyan')
        labels = dict(face='Gear Face', root_cut='Root Cut', dropcut='Drop Cut', root_arc='Root Arc', tip_arc='Tip Arc')
        seen = set()
        parts = g.gen_gear_tooth_parts(closed=True, include_extras=True)
        extras = dict(parts)
        # import pprint; pprint.pp(extras.keys())

        plot(circle(g.base_radius), 'wheat')

        gear_face: Involute = extras['_gear_face']
        gear_face.end_angle *= 2
        gear_face.start_angle = 0
        plot(gear_face.path(20), 'green')

        root_cut: InvoluteWithOffsets = extras['_root_cut']
        root_cut.end_angle *= 2
        root_cut.start_angle *= 2
        plot(root_cut.path(20), 'blue')

        intersection: Point = extras['_face_root_intersection']
        # plot(cross(0.1, intersection), 'pink')
        # plot(circle(0.1, intersection), 'pink')
        plt.annotate(
            'Intersection of gear face\ninvolute and root cut trochoid', intersection.xy(),
            xytext=(-10, 40), textcoords='offset points',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->'),
            bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha=0.8),
        )

        if False:
            for tag, points in parts:
                if tag.startswith('_'):
                    continue
                plot(points, color=colors[tag], linewidth=3, label=labels[tag] if tag not in seen else None)
                seen.add(tag)
                if tag == 'root_arc':
                    # Plot this again mirrored so that plot is symmetric
                    plot([(p.x, -p.y) for p in points], color=colors[tag], linewidth=3)
            plt.legend(loc=6)
        g.plot_show((3, 1, 2))


def main():
    # doc_radii()
    # doc_tooth_parts()
    doc_tooth_equations()


if __name__ == '__main__':
    main()
