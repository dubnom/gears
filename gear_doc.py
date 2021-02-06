"""
Various plots to document gear calculations
See also: https://www.tec-science.com/mechanical-power-transmission/involute-gear/calculation-of-involute-gears/
"""
import os
from math import cos, sin, radians, degrees, tau, tan
from typing import Union, Optional, Tuple

import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend, DraggableLegend
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from x7.geom.geom import PointList, Point, Vector, XYList, XYTuple, PointUnionList, BasePoint
from x7.geom.utils import circle, min_rotation, path_rotate_ccw, path_from_xy
from x7.geom.plot_utils import plot
from x7.lib.iters import t_range, iter_rotate
from gear_involute import GearInvolute, InvoluteWithOffsets, InvolutePair
from rack import Rack

GEAR_COLOR = 'lightgrey'
GEAR_PS_COLOR = 'darkgrey'
RACK_COLOR = '#80CCFF'
RACK_PS_COLOR = '#6099BF'
RACK_CUTTER_COLOR = '#FF8080'

IMAGES_GEAR_DOC = './doc/images/gear_doc'


def plot_show(fig_name: str, gear: Optional[GearInvolute], zoom_radius: Union[None, float, tuple],
              axis='x', grid=False, title_loc='tc'):
    fig: Figure = plt.gcf()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.set_window_title(fig_name+'.png')

    ax: Axes = plt.gca()
    ax.axis('off')
    title = ax.title.get_text()
    # title = 'y' + title + 'j'
    if title_loc == 'tc':
        title_loc = (0.5, 0.95)
    elif title_loc == 'tl':
        title_loc = (0.2, 0.95)
    elif title_loc == 'tr':
        title_loc = (0.8, 0.95)
    title_a = plt.annotate(
        title, title_loc, xycoords='figure fraction',
        bbox=dict(facecolor='#EEEEEE', edgecolor='none', pad=6.0, alpha=1),
        fontsize=ax.title.get_fontsize(),
        fontfamily=ax.title.get_fontfamily(),
        ha='center', va='top',
        fontproperties=ax.title.get_fontproperties(),
    )
    title_a.draggable()

    if gear:
        gear.plot_show(zoom_radius, axis=axis, grid=grid)
    else:
        plt.axis('equal')
        if grid:
            plt.grid()
        # Set zoom_radius to zoom in around where gears meet
        if zoom_radius:
            zxl, zxr, zy = zoom_radius if isinstance(zoom_radius, tuple) else (zoom_radius, zoom_radius, zoom_radius)
            ax = plt.gca()
            ax.set_xlim(-zxl, zxr)
            ax.set_ylim(-zy, zy)
        plt.show()

    out = os.path.join(IMAGES_GEAR_DOC, fig_name+'.png')
    print('Saving to', out)
    fig.savefig(out)


def plot_fill(xy: PointUnionList, color='black', plotter=None, label=None) -> Artist:
    """Quick entry to pyplot.fill() for List[Point]"""

    plotter = plotter or plt
    artist, = plotter.fill(*zip(*xy), color)

    if label:
        artist.set_label(label)
    return artist


def ha_va_from_xy(xytext) -> Tuple[str, str]:
    x, y = xytext
    ha = 'left' if x > 0 else 'right' if x < 0 else 'center'
    va = 'bottom' if y > 0 else 'top' if y < 0 else 'center'
    return ha, va


def plot_annotate(text, where, xytext, xycoords='data', noarrow=False) -> Annotation:
    if isinstance(xytext, str):
        v, h = xytext[0], xytext[1]
        h = {'c': 0, 'r': 1, 'l': -1}[h]
        v = {'c': 0, 'u': 1, 'd': -1}[v]
        xytext = ((25 if v == 0 else 10) * h, 20 * v)
    if xytext:
        ha, va = ha_va_from_xy(xytext)
    else:
        ha = 'center'
        va = 'center'
        xytext = (0, 0)
    arrowprops = None if noarrow or xytext == (0, 0) else dict(arrowstyle='->')

    if isinstance(where, (BasePoint, Vector)):
        where = where.xy()
    elif isinstance(where, Line2D):
        x_data = where.get_xdata()
        y_data = where.get_ydata()
        where = (x_data[0]+x_data[-1])/2, (y_data[0]+y_data[-1])/2
    annotation = plt.annotate(
        text, where, xycoords=xycoords,
        xytext=xytext, textcoords='offset points',
        ha=ha, va=va,
        arrowprops=arrowprops,
        bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha=0.8),
    )
    annotation.draggable()
    return annotation


def path_close(path: PointList) -> PointList:
    return path + [path[0]]


def path_len(path: PointList) -> float:
    total = 0
    a: Point
    b: Point
    for a, b in iter_rotate(path, 2, cycle=False):
        total += (a - b).length()
    return total


def arrow(path: PointList, where='end', tip_len=None) -> PointList:
    """
        Attach arrowheads to a path

        :param path: input path
        :param where: where to put arrowheads: None->no arrow, 'both', 'start', 'end'
        :param tip_len: length of arrowhead tip: None->auto calc
        :return: PointList
    """

    if tip_len is None:
        tip_len = path_len(path) / 10
    head = []
    tail = []
    if where in ('both', 'end'):
        end_pt = Point(*path[-1])
        v = Point(*path[-2]) - end_pt
        v = v.unit() * tip_len
        vl = v.rotate(15)
        vr = v.rotate(-15)
        tail = [end_pt+vl, end_pt, end_pt+vr]
    if where in ('both', 'start'):
        end_pt = Point(*path[0])
        v = Point(*path[1]) - end_pt
        v = v.unit() * tip_len
        vl = v.rotate(15)
        vr = v.rotate(-15)
        head = [end_pt+vl, end_pt, end_pt+vr]
    if where not in (None, 'none', 'start', 'end', 'both'):
        raise ValueError('where: must be None, start, end, or both, not %r' % arrow)
    return head + path + tail


# noinspection PyShadowingNames
def arc(r, sa, ea, c: BasePoint = Point(0, 0), steps=-1, direction='shortest',
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

    arc_len = abs(radians(ea - sa)) * r
    arrow_len = min(arc_len / 3, r / 4)
    arc_prev = min(5, len(path)-1)
    if arrow in ('both', 'start'):
        end_pt = Point(*path[0])
        v = Point(*path[arc_prev]) - end_pt
        v = v.unit() * arrow_len
        vl = v.rotate(15)
        vr = v.rotate(-15)
        path = [end_pt+vl, end_pt, end_pt+vr] + path
    if arrow in ('both', 'end'):
        end_pt = Point(*path[-1])
        v = Point(*path[-arc_prev]) - end_pt
        v = v.unit() * arrow_len
        vl = v.rotate(15)
        vr = v.rotate(-15)
        path.extend([end_pt+vl, end_pt, end_pt+vr])
    if arrow not in (None, 'none', 'start', 'end', 'both'):
        raise ValueError('arrow: must be None, start, end, or both, not %r' % arrow)
    if mid:
        t = radians(sa+ea) / 2
        mid = (r * cos(t), r * sin(t))
        return path, mid
    else:
        return path


# noinspection PyShadowingNames
def angle(text: str, center: BasePoint, p1: Union[Vector, BasePoint],
          p2: Union[Vector, BasePoint, None] = None, degrees: Optional[float] = None,
          xytext: Optional[XYTuple] = None, vec_len: Optional[float] = None, arc_pos=None,
          color='cyan', direction=None,
          arc_color=None, vec_color=None, arrow: Optional[str] = 'end',
          ) -> Tuple[Annotation, Artist, Artist]:
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
        :param degrees: Angle to second point (exclusive with p2)
        :param xytext:  Override default xytext location
        :param vec_len:    Override displayed p1 & p2 vector lengths
        :param arc_pos: Position of angular arc along vectors (default 0.5)
        :param color:   Color of arc & vectors
        :param arc_color: Color of arc (defaults to color, 'none' for no arc)
        :param vec_color: Color of vectors (defaults to color, 'none' for no vectors)
        :param arrow:     'start', 'end', 'both', 'none' or None
        :param direction: Direction of arc (ccw, cw, shortest).  None defaults to shortest
        :return: (Annotation, vectors Artist, arc Artist)
    """
    p1v = p1 if isinstance(p1, Vector) else (p1 - center)
    if p2 is None:
        if degrees is None:
            # degrees is required
            raise ValueError('One of p2 or degrees must be supplied')
        p2v = p1v.rotate(degrees)
        if direction is None:
            direction = 'ccw' if degrees > 0 else 'cw'
    else:
        if degrees is not None:
            raise ValueError('Only one of p2 or degrees can be supplied')
        p2v = p2 if isinstance(p2, Vector) else (p2 - center)
    arc_color = arc_color or color
    vec_color = vec_color or color
    vec_len = vec_len if vec_len else max(p1v.length(), p2v.length())
    vec_artist = plot([center + p1v.unit() * vec_len, center, center + p2v.unit() * vec_len], color=vec_color)
    if vec_color == 'none':
        vec_artist.remove()
    if degrees is None:
        degrees = (p2v.angle() - p1v.angle()) % 360
    text = text.replace('@', r'${angle:.1f}^\circ$')
    text = text.format(angle=degrees, at='@')
    degrees = abs(degrees)

    if arc_pos is None:
        if degrees < 15:
            arc_pos = 0.95
        elif degrees < 40:
            arc_pos = 0.75
        else:
            arc_pos = 0.5
    arc_path, midv = arc(arc_pos * vec_len, p1v.angle(), p2v.angle(), center, direction=direction, arrow=arrow, mid=True)
    arc_artist = plot(arc_path, color=arc_color)
    if arc_color == 'none':
        arc_artist.remove()
    midv = Vector(*midv)
    # plot([center, center + midv])
    xo = -1 if midv.x < 0 else 1
    yo = -1 if midv.y < 0 else 1
    xytext = (10*xo, 20*yo) if xytext is None else xytext
    ha, va = ha_va_from_xy(xytext)

    annotation = plt.annotate(
        text, (center+midv).xy(),
        xytext=xytext, textcoords='offset points',
        ha=ha, va=va,
        arrowprops=dict(arrowstyle='->'),
        bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha=0.8),
    )
    annotation.draggable()
    return annotation, vec_artist, arc_artist


def doc_radii():
    save_legend = []

    def legend2():
        line = plot([(0, 0)], color='white')
        leg1 = plt.legend(loc=3)
        save_legend.append(DraggableLegend(leg1))
        leg2: Legend = plt.legend([line], [extra], loc=(0.01, 0.6))
        leg2.set_title('Inputs')
        save_legend.append(DraggableLegend(leg2))
        plt.gca().add_artist(leg1)

    rot = -0.35

    g = GearInvolute(17, profile_shift=0, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot_fill(g.rack().path(teeth=5, rot=g.rot+0.5, closed=True), color=RACK_COLOR)
    extra = g.plot(pressure_line=False, color='fill-'+GEAR_COLOR)
    plt.title('Involute Radii')
    legend2()
    plot_show('inv_radii', g, (5.5, 1.5, 3))

    g = GearInvolute(17, profile_shift=0.5, rot=rot, **GearInvolute.HIGH_QUALITY)
    plot_fill(g.rack().path(teeth=5, rot=g.rot+0.5, closed=True), color=RACK_COLOR)
    extra = g.plot(pressure_line=True, color='fill-'+GEAR_COLOR)
    plt.title('Involute Radii with Profile Shift of 0.5')
    legend2()
    plot_show('inv_radii_ps', g, (6, 1, 3))


def doc_tooth_parts():
    for teeth in [7, 27]:
        g = GearInvolute(teeth, **GearInvolute.HIGH_QUALITY)
        g.curved_root = False
        plot_fill(g.instance().poly, 'lightgrey')
        plt.title('Tooth path for %d teeth' % g.teeth)
        plot_fill(g.rack().path(5, rot=0.5, closed=True), RACK_COLOR)
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
        plot_show('inv_parts_%d' % teeth, g, (3, 1, 2))


def test_angle():
    # plot(arc(5, 45, 45+270, Point(2, 2)), color='cyan')
    angle('Something is @', Point(-20, 0), Vector(10, 1), Vector(8, 3), color='green', arc_pos=0.95, arrow='both')
    angle(r'$\alpha$ is @', Point(-20, 0), Vector(8, 4), Vector(-8, 4), color='green', arc_pos=0.95)
    angle('Hi-precision is {angle:8.5f}', Point(-20, 0), Vector(8, 4), Vector(-8, 4), color='green', arc_pos=0.5)
    angle('Something Blue', Point(0, 0), Point(1, 0), Point(0, 1), vec_len=15, color='blue')
    angle('Something Else', Point(20, 0), Vector(10, 1), degrees=-270, arrow=None)
    angle('15,0 < 0,15', Point(0, 20), Vector(5, 0), Vector(0, 5), color='red', direction='ccw')
    angle('0,15 < 15,0', Point(0, -20), Vector(0, 5), Vector(5, 0), color='pink', direction='ccw')
    angle('ang 10: @', Point(-30, -20), Vector(10, 10), degrees=10, color='pink', arc_color='red', direction='ccw')
    angle('ang 30{at}@', Point(-30, -20), Vector(10, 10), degrees=30, color='pink', vec_color='red', xytext=(1, -40), direction='ccw')
    angle('ang 100: @', Point(-30, -20), Vector(10, 10), degrees=100, color='pink', direction='ccw')
    angle('Just Arc', Point(20, -20), Vector(10, 10), degrees=90, color='pink', vec_color='none', direction='ccw', arrow='both')
    angle('Just Vecs', Point(20, -20), Vector(-10, -10), degrees=90, color='pink', arc_color='none', direction='ccw')
    plot([(0, -30)], color='white')
    plt.axis('equal')
    plt.grid()
    plt.show()


def doc_tooth_equations():
    # pitch
    # tooth
    # half_tooth
    doc_tooth_equations_fig('rack_dim', 'Rack Dimensions', 3, 'rack_0.5 pitch_line pitch_line_a pitch tooth half_tooth')
    doc_tooth_equations_fig('rack_cut', 'Rack (with Cutter) and Gear', 4, 'rack_tall cutter gear rack_a gear_a')
    doc_tooth_equations_fig('tip_path', 'Path of the Cutter Tip Trochoid', 3, 'rack_tall gear trochoid_0')

    # undercut aka root_cut, face-root intersection
    doc_tooth_equations_fig('intersection',
                            'Intersection of Gear Face Involute and Cutter Tip Trochoid\n'
                            '(with teeth=5 for elucidation)',
                            1.5,
                            'base_radius rack_tall gear gear_face_full trochoid_0 rf_intersection',
                            teeth=5)

    # pp_inv_angle
    # pp_off_angle
    doc_tooth_equations_fig('calc_offset', 'Calculation of Involute offset', 3,
                            'radii radii_a gear gear_face gear_face_a pp involute_0')

    # half_tooth_ps
    doc_tooth_equations_fig('ps', 'Profile Shift',
                            1.5,
                            'pitch_radius gear_o gear_ps_fill gear_ps_a',
                            teeth=11, title_loc='tr')
    doc_tooth_equations_fig('ps_calc',
                            'Profile Shift',
                            2,
                            'rack_o rack_ps pitch_line pitch_radius gear_o gear_ps_fill half_tooth_ps ps_legend',
                            teeth=11, title_loc='tl')

    doc_tooth_equations_fig('all', 'All fields', 5, 'all')


def doc_tooth_equations_fig(fig_name, title, zoom, fig, teeth=27, title_loc='tc'):
    class FigAll:
        found = set()

        def __contains__(self, item):
            self.found.add(item)
            return True

    if fig == 'all':
        fig = FigAll()
    elif isinstance(fig, str):
        fig = set(fig.split())

    pressure_angle = 20

    ang = 90

    g = GearInvolute(teeth, module=1, pressure_angle=pressure_angle, **GearInvolute.HIGH_QUALITY)
    rack = g.rack(angle=ang)

    g_ps = GearInvolute(teeth, module=1, pressure_angle=pressure_angle, profile_shift=0.4, **GearInvolute.HIGH_QUALITY)
    rack_ps = g_ps.rack(angle=ang)

    extra_rot = 0.5
    base_rot = g.teeth / 4 - extra_rot
    base_ang = base_rot / g.teeth * 360
    center = Point(0, 0)
    zero_vec = Vector(1, 0).rotate(ang)
    parts = g.gen_gear_tooth_parts(closed=True, include_extras=True)
    extras = dict(parts)
    localz: dict = extras['_locals']

    plt.title(title)
    """
    pitch = M * pi
    pitch_radius = P_r = M * T / 2
    tooth_width = T_w = pitch / 2 = M * pi / 2
    
    pitch_angle = P_a = 2 * pi / T
    tooth_angle = T_a = P_a / 2 = pi / T
    half_tooth_angle = HT_a = P_a / 4
    
    T_w = T_a * P_r = pi / T * M * T / 2 = pi * M / 2 
    """

    def rack_line(r, l):
        r = 2 * g.pitch_radius - r
        return path_rotate_ccw([Point(r, -l/2), Point(r, l/2)], ang, as_pt=True)

    rack_rot = -extra_rot
    pitch_line = rack_line(g.pitch_radius, 200)
    pitch_segment = rack_line(g.tip_radius + 0.5, g.pitch)
    tooth_segment = rack_line(g.pitch_radius, g.pitch/2)
    if 'rack' in fig:
        rack_path = rack.path(teeth=7, rot=rack_rot, closed=True)
        plot_fill(rack_path, color=RACK_COLOR)
    if 'rack_ps' in fig:
        rack_path = rack_ps.path(teeth=7, rot=rack_rot, closed=True)
        plot_fill(rack_path, color=RACK_COLOR)
    if 'rack_o' in fig:
        rack_path = rack.path(teeth=7, rot=rack_rot, closed=True)
        plot(rack_path, color=RACK_PS_COLOR, linestyle=':')
    if 'rack_tall' in fig:
        rack_tall = g.rack(angle=ang, tall_tooth=True)
        rack_path = rack_tall.path(teeth=7, rot=rack_rot, closed=True)
        plot_fill(rack_path, color=RACK_CUTTER_COLOR)
        rack_path = rack.path(teeth=7, rot=rack_rot, closed=True)
        plot_fill(rack_path, color=RACK_COLOR)
        if 'cutter' in fig:
            rack_path = rack_tall.path(teeth=1, rot=rack_rot)
            mid = rack_path[1].mid(rack_path[2])
            plot_annotate('Cutter', mid + Vector(0, 0.1), 'dc')
        # plot(rack_path, color='#6099BF', linestyle=':')
    if 'rack_a' in fig:
        plot_annotate('Rack', Vector(g.tip_radius+1.5, 0).rotate(ang), 'cc')
    if 'rack_0.5' in fig:
        rack_path = rack.path(teeth=7, rot=rack_rot+0.5, closed=True)
        for side in [-1, 1]:
            cl = path_rotate_ccw([Point(0, side*g.pitch/2), Point(100, side*g.pitch/2)], ang)
            plot(cl, 'lightgrey')
        plot_fill(rack_path, color=RACK_COLOR)
    if 'pitch_line' in fig:
        plot(pitch_line, 'lightgreen')
        if 'pitch_line_a' in fig:
            plot_annotate('Pitch Line', zero_vec * g.pitch_radius + Vector(g.pitch / 2, 0), 'cc')
    if 'pitch' in fig:
        plot(arrow(pitch_segment, 'both'), 'green')
        plot_annotate('Pitch', pitch_segment[0].mid(pitch_segment[1]), 'dc')
    if 'tooth' in fig:
        plot(arrow(tooth_segment, 'both'), 'green')
        plot_annotate('Tooth\n=\nPitch / 2', tooth_segment[0].mid(tooth_segment[1]), 'uc')

    if 'gear_ps' in fig:
        plot(path_close(g_ps.instance().poly_at(base_rot)), GEAR_PS_COLOR)
    if 'gear_ps_fill' in fig:
        plot_fill(path_close(g_ps.instance().poly_at(base_rot)), GEAR_PS_COLOR)
    if 'gear' in fig:
        plot_fill(path_close(g.instance().poly_at(base_rot)), GEAR_COLOR)
    if 'gear_o' in fig:
        plot(path_close(g.instance().poly_at(base_rot)), GEAR_COLOR, linestyle=':')
    if 'gear_a' in fig:
        plot_annotate('Gear', Vector(g.root_radius-1.0, 0).rotate(ang), 'cc')
    if 'gear_ps_a' in fig:
        a = zero_vec * g.tip_radius
        b = zero_vec * g_ps.tip_radius      # TODO-what about tip shortening?
        plot_annotate('Profile Shift', plot([a, b], 'green'), 'cl')

    radii_config = [
        ('root_radius', g.root_radius, 'Root Radius', 'green', 3),
        ('base_radius', g.base_radius, 'Base Radius', 'wheat', 2),
        ('pitch_radius', g.pitch_radius, 'Pitch Radius', 'lightgreen', 1),
        ('tip_radius', g.tip_radius, 'Tip Radius', 'green', 0),
    ]
    for key, val, label, color, ang_shift in radii_config:
        if 'radii' in fig or key in fig:
            plot(circle(val), color)
            loc = zero_vec.rotate(8 + ang_shift) * val + Vector(-0.5, -0.1)
            if 'radii_a' in fig or (key+'_a') in fig:
                plot_annotate(label, loc, 'cc')

    if 'half_tooth_ps' in fig:
        rack_path = rack.path(teeth=1, rot=rack_rot, closed=True)
        mid = rack_path[1].mid(rack_path[2])
        rack_path_ps = rack_ps.path(teeth=1, rot=rack_rot, closed=True)
        mid_ps = rack_path_ps[1].mid(rack_path_ps[2])
        l = plot([mid, mid_ps], 'green')
        plot_annotate('Profile Shift: $x$', l, (-25, -35))

        tw_mid = Point(0, g.pitch_radius)
        tw_half = Vector(localz['half_tooth'], 0)
        l = plot([tw_mid-tw_half, tw_mid+tw_half], 'red')
        plot_annotate('Tooth Width', l, 'dc')

        tw_ps = Vector(g_ps.profile_shift * g_ps.module * tan(radians(pressure_angle)), 0)
        tw_ps_l = tw_mid + tw_half
        tw_ps_r = tw_ps_l + tw_ps
        l = plot([tw_ps_l, tw_ps_r], 'green')
        plot_annotate('Extra Width due to Profile Shift\n'
                      r'$x \cdot \tan \alpha$',
                      l, 'uc')
        angle('Pressure Angle\n'
              r'$\alpha$', rack_path[1], rack_path_ps[1], rack_path_ps[1]-tw_ps,
              arrow='none', xytext=(0, 25))

    pointer = arrow([center, Point(g.pitch_radius_effective, 0)], tip_len=0.5 * g.module)
    pointer = path_rotate_ccw(pointer, base_ang)
    if 'pointer' in fig:
        plot(pointer, color='wheat')

    if 'pp' in fig:
        pp_vec = zero_vec.rotate(degrees(localz['pp_off_angle'])) * g.pitch_radius_effective
        angle('pp_off_angle', center, zero_vec, pp_vec,
              vec_len=g.pitch_radius_effective * 2, arc_pos=0.52, arrow='none')
        plot_annotate('(ppx, ppy)', pp_vec, 'dl')
        # print(localz['pp_inv_angle'], localz['pp_off_angle'])

    gear_face: InvoluteWithOffsets = extras['_gear_face']
    if 'gear_face' in fig:
        path = path_rotate_ccw(path_from_xy(gear_face.path(30)), ang)
        plot(path, 'green')
        p = path[0]
        for p in path:
            if (Point(*p) - center).length() > (g.tip_radius-0.1 + g.pitch_radius) * 0.5:
                break
        if 'gear_face_a' in fig:
            plot_annotate('Involute offset by\npp_off_angle + half_tooth', p, (20, 0))
            angle('half_tooth', center, zero_vec, degrees=-90/g.teeth,
                  xytext=(0, -15),
                  arc_pos=g.pitch_radius, arrow='both', vec_color='none')

    if 'gear_face_full' in fig:
        gff = gear_face.copy()
        gff.end_angle *= 2
        gff.start_angle = 0
        path = path_rotate_ccw(path_from_xy(gff.path(30)), ang)
        plot(path, 'green')

    if 'involute_0' in fig:
        save_end = gear_face.end_angle
        save_start = gear_face.start_angle
        gear_face.end_angle *= 2
        gear_face.start_angle = 0
        oa = degrees(gear_face.offset_angle)
        path = path_rotate_ccw(path_from_xy(gear_face.path(30)), ang - oa)
        plot(path, 'green')
        p = path[0]
        for p in path:
            if (Point(*p) - center).length() > g.tip_radius + 0.5 * g.module:
                break
        plot_annotate('Involute @ zero offset', p, (20, 0))
        # print(p)
        gear_face.end_angle = save_end
        gear_face.start_angle = save_start

    root_cut: InvoluteWithOffsets = extras['_root_cut']
    if 'trochoid_0' in fig:
        trochoid_0 = root_cut
        # trochoid_0.offset_radius = localz['addendum_ps']
        # trochoid_0.offset_norm = localz['addendum_offset']
        trochoid_0.end_angle = tau
        trochoid_0.start_angle = -tau
        plot(path_rotate_ccw(trochoid_0.path_pt(200), ang), 'blue')

    root_cut.end_angle *= 2
    root_cut.start_angle *= 2
    plot(root_cut.path(20), 'blue')

    if 'rf_intersection' in fig:
        intersection: Point = extras['_face_root_intersection']
        loc = (intersection - center).rotate(ang)
        # plot(cross(0.1, intersection), 'pink')
        # plot(circle(0.1, intersection), 'pink')
        plot_annotate('Intersection', loc, 'cl')
        plt.annotate(
            'Intersection of gear face\ninvolute and root cut trochoid', intersection.xy(),
            xytext=(40, -80), textcoords='offset points',
            ha='center', va='top',
            arrowprops=dict(arrowstyle='->'),
            bbox=dict(facecolor='white', edgecolor='darkgrey', pad=4.0, alpha=0.8),
        )

    if 'ps_legend' in fig:
        plot_annotate(
            'dotted: original\n'
            'solid:  shift of 0.4',
            (0.2, 0.7),
            (0, 0),
            xycoords='figure fraction',
            noarrow=True,
        )

    plot_show('inv_' + fig_name, g, zoom * g.module,
              axis='y' if ang == 90 else 'x', title_loc=title_loc)
    if isinstance(fig, FigAll):
        print('Found:', ', '.join(sorted(fig.found)))


def doc_intro():
    plt.gcf().set_size_inches(3, 2)
    g = InvolutePair(25, 9)
    plot_fill(g.pinion().poly_at(), GEAR_PS_COLOR)
    plot_fill(g.wheel().poly_at(), GEAR_COLOR)
    gp = g.gear_pinion
    plot_fill(gp.rack().path(9, closed=True, depth=4), RACK_COLOR)
    plot_show('intro_gears', g.gear_wheel, None)


def doc_rack():
    plt.title('Rack Calculations')
    r = Rack(angle=-90)
    plot_fill(r.path(5, closed=True, depth=3), RACK_COLOR, label=r'$\alpha = 20_\circ$')

    plot(r.pitch_line(5), 'grey', linestyle=':')
    plot_annotate('Pitch Line', (-r.circular_pitch, 0), 'cc')
    plot(r.tip_line(5), 'grey', linestyle=':')
    plot_annotate('Tip Line', (-r.circular_pitch, -r.tip_radius), 'cc')
    plot(r.root_line(5), 'grey', linestyle=':')
    plot_annotate('Root Line', (-r.circular_pitch, -r.root_radius), 'cc')

    path = r.path(1)
    pv = Point(path[0].x, path[1].y)
    angle(r'$\alpha$: @', path[0], pv, path[1], color='blue', arc_pos=0.6, vec_len=3, arrow='none', xytext=(50, -10))
    l = plot([pv, path[1]], 'red')
    plot_annotate(r'$H_t * \tan \alpha$', l, 'uc')
    l = plot([pv, path[0]], 'green')
    plot_annotate(r'$H_t$: total', l, 'cr')

    cl = -r.circular_pitch * 3 / 4
    l = plot([(cl, 0), (cl, -r.tip_radius)], 'green')
    plot_annotate('$H_a$: addendum', l, 'cl')
    l = plot([(cl, 0), (cl, -r.root_radius)], 'cyan')
    plot_annotate('$H_d$: dedendum', l, 'cl')
    l = plot([(cl, -r.tip_radius), path[2]], 'red')
    plot_annotate(r'$H_a * \tan \alpha$', l, 'uc')

    plot_show('inv_rack_calcs', None, (1.5 * r.circular_pitch, 0.5 * r.circular_pitch, 1))

    plt.title('Rack Pressure Angles')
    r = Rack(angle=-90)
    plot_fill(r.path(5, closed=True, depth=3), RACK_COLOR, label=r'$\alpha = 20_\circ$')
    r = Rack(angle=-90, pressure_angle=14.5)
    plot(path_close(r.path(5, closed=True, depth=3)), 'red', label=r'$\alpha = 14.5_\circ$')
    r = Rack(angle=-90, pressure_angle=30)
    plot(path_close(r.path(5, closed=True, depth=3)), 'green', label=r'$\alpha = 30_\circ$')
    plt.legend()
    plot_show('inv_rack_angles', None, (1.5 * r.circular_pitch, 0.5 * r.circular_pitch, 1))


def main():
    # test_angle()
    doc_rack()
    doc_intro()
    doc_radii()
    doc_tooth_parts()
    doc_tooth_equations()


if __name__ == '__main__':
    main()
