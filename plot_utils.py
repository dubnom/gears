from math import cos, sin, radians
from typing import Union, Tuple, Optional

from matplotlib import pyplot as plt
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
from x7.geom.geom import Point, PointUnionList, BasePoint, Vector, PointList, XYList, XYTuple
from x7.geom.plot_utils import plot
from x7.geom.utils import min_rotation
from x7.lib.iters import iter_rotate, t_range


class PlotZoomable:
    """Base class for plot_zoom behavior"""

    ZoomRadiusType = Union[None, float, Tuple[float, float, float], Tuple[float, float, float, float]]
    # no-zoom, all, (l, r, u+d), (l, r, u, d),

    @property
    def plot_zoom_pos(self) -> float:
        """Radial distance to zoom_point"""
        return getattr(self, 'pitch_radius_effective', 0.0)

    def plot_zoom_point(self, axis='x') -> Point:
        if axis == 'x':
            return Point(self.plot_zoom_pos, 0)
        elif axis == '-x':
            return Point(-self.plot_zoom_pos, 0)
        elif axis == 'y':
            return Point(0, self.plot_zoom_pos)
        elif axis == '-y':
            return Point(0, -self.plot_zoom_pos)
        else:
            raise ValueError('axis, expected "x", "y", "-x", or "-y", not %r' % axis)

    @property
    def plot_zoom_scale(self) -> float:
        """Multiply zoom_radius by this (usually gear.module)"""
        return getattr(self, 'module', 1.0)

    def plot_zoom_to(self, zoom_radius: ZoomRadiusType, axis='x', plotter=None):
        """Set x/y limits to zoom in around zoom_point"""
        ax = plotter or plt.gca()
        if zoom_radius:
            c = self.plot_zoom_point(axis=axis)
            if isinstance(zoom_radius, tuple):
                if len(zoom_radius) == 4:
                    zxl, zxr, zyu, zyd = zoom_radius
                elif len(zoom_radius) == 3:
                    zxl, zxr, zyu, zyd = zoom_radius + (zoom_radius[2], )
                else:
                    raise ValueError('zoom_radius tuple must be len 3 or 4, not %d' % len(zoom_radius))
            else:
                zxl, zxr, zyu, zyd = zoom_radius, zoom_radius, zoom_radius, zoom_radius
            sc = self.plot_zoom_scale
            zxl, zxr, zyu, zyd = sc*zxl, sc*zxr, sc*zyu, sc*zyd

            if axis in ('x', '-x'):
                ax.set_xlim(c.x - zxl, c.x + zxr)
                ax.set_ylim(c.y - zyd, c.y + zyu)
            else:
                ax.set_xlim(c.x - zyd, c.x + zyu)
                ax.set_ylim(c.y - zxl, c.y + zxr)

    def plot_show(self, zoom_radius: ZoomRadiusType = None, axis='x', grid=True, plotter=None):
        """Set axis equal, display grid, zoom, and show.  If plotter!=None, show() is not done."""
        ax: Axes = plotter or plt.gca()
        ax.axis('equal')
        if grid:
            ax.grid()
        self.plot_zoom_to(zoom_radius, axis=axis, plotter=ax)
        if not plotter:
            plt.show()


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


def main():
    test_angle()


if __name__ == '__main__':
    main()
