from math import sin, cos, tau, radians
from typing import List, Union, Tuple

from matplotlib import pyplot as plt

from anim.geom import BasePoint, Point, Vector, XYList
from anim.transform import Transform

__all__ = [
    'PointList', 'PointUnionList', 'check_point_list',
    't_range',
    'path_rotate', 'path_translate', 'circle', 'arc', 'plot',
    'GearInstance',
]
PointList = List[BasePoint]
PointUnionList = List[Union[BasePoint, float, Tuple[float, float]]]


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


class GearInstance:
    """Holder for finished gear (could be just about any polygon, really)"""
    def __init__(self, module, teeth, poly: PointList, center: Point):
        self.module = module
        self.teeth = teeth
        self.poly = poly    # center of poly is 0, 0
        check_point_list(poly)
        self.center = center
        # self.circular_pitch = self.module * pi
        self.pitch_diameter = self.module * self.teeth
        self.pitch_radius = self.pitch_diameter / 2

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
        plot(path_translate(path_rotate(self.poly, rotation, True), self.center), color, plotter=plotter)

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
