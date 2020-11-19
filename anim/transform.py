from abc import ABC, abstractmethod

import numpy as np
import math
from .typing import *
from .geom import *


__all__ = ['EmptyStackError', 'Transform', 'Transformer', 'Matrix', 'NumpyArray', 'NumpyPoints']

NumpyArray = np.ndarray
NumpyPoints = np.ndarray    # shape==(3, N)


def near_zero(v):
    return abs(v) < 1e-16


class EmptyStackError(Exception):
    pass


class Matrix(object):
    """Just a transformation matrix."""
    def __init__(self, mat3x3=None):
        self.ndmat = np.identity(3) if mat3x3 is None else np.array(mat3x3, dtype=float)

    def __str__(self):
        return str(tuple(tuple(round(v, 2) for v in row) for row in self.as_tuple3x3()))

    def __repr__(self):
        return 'Transform(%s)' % repr(self.as_tuple3x3())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.as_tuple3x3() == other.as_tuple3x3()

    def as_tuple3x3(self):
        return tuple(tuple(row) for row in self.ndmat)

    def copy(self):
        return Matrix(self.ndmat)

    def pre_dot_eq(self, ndmat):
        """self = mat _dot_ self"""
        self.ndmat = np.dot(ndmat, self.ndmat)

    def post_dot_eq(self, ndmat):
        """self = self _dot_ mat"""
        self.ndmat = np.dot(self.ndmat, ndmat)

    def inverse(self):
        """Calculate inverse"""
        return Matrix(np.linalg.inv(self.ndmat))


class Transformer(ABC):
    """
        Base class for transformation operations.

        Note: Numpy Points are represented as np.array([(x, y, 1), ...]).T
    """

    LINEAR = True

    @staticmethod
    def points_to_np_points(points: List[BasePoint]) -> NumpyPoints:
        """Convert list of Points into multiple numpy points"""
        return np.array([(x, y, 1) for x, y in points]).T

    @staticmethod
    def xy_to_np_point(x, y) -> NumpyPoints:
        """Convert single point into single numpy point"""
        return np.array([(x, y, 1)]).T

    @abstractmethod
    def transform_np_pts(self, np_pts: NumpyPoints):
        """
            Transform numpy points to numpy points.

            :param np_pts:
            :return: either NumpyArray or [(x, y, 1), ...]
        """
        ...

    def transform(self, x, y):
        """Transform (x, y) to (x, y)"""
        pp = self.transform_np_pts(self.xy_to_np_point(x, y))[0]
        return pp[0], pp[1]

    def transform_width(self, width):
        """Scale width by average of x & y scales"""
        width *= math.sqrt(2)/2
        zx, zy = self.transform(0, 0)
        wx, wy = self.transform(width, width)
        return math.hypot(wx-zx, wy-zy)

    def transform_pt(self, pt: BasePoint, as_pt=True) -> Point:
        """Transform a Point.  Return either (x, y) or Point(x, y)"""
        xy = self.transform(pt.x, pt.y)
        return Point(*xy) if as_pt else xy

    def transform_pts(self, pts: List, flatten=False):
        """
            Transform a list.  Returns [x, y, ...], [(x, y), ...], or [Point(x, y), ...]
            :param pts: List of Point or (x, y) tuples
            :param flatten: True to return [x, y, ...].  False to return tuples or Points, depending on input
            :return:
        """
        np_pts = self.transform_np_pts(self.points_to_np_points(pts))
        if flatten:
            return [v for x, y, z in np_pts for v in (x, y)]
        elif isinstance(pts[0], BasePoint):
            return [Point(x, y) for x, y, z in np_pts]
        else:
            return [(x, y) for x, y, z in np_pts]


class Transform(Transformer):
    def __init__(self, mat2x3=None):
        # Note: .mat contents must be treated as read-only since inverse is cached
        if mat2x3:
            m = [mat2x3[0:3], mat2x3[3:6], [0, 0, 1]]
            self._mat = Matrix(m)
        else:
            self._mat = Matrix()
        self._inverse = None
        self.stack = []

    def __str__(self):
        return str(tuple(round(v, 2) for v in self.as_6tuple()))

    def __repr__(self):
        return 'Transform(%s)' % repr(self.as_6tuple())

    def __eq__(self, other):
        # TODO-should this care about stack and _inverse? (would have to fix tests)
        return isinstance(other, Transform) and self._mat == other._mat

    def as_6tuple(self):
        return tuple(v for row in self._mat.ndmat[:2] for v in row)

    def copy(self):
        dup = Transform(self.as_6tuple())
        dup.set_matrix(self)
        return dup

    def set_matrix(self, xform: 'Transform'):
        self._mat = xform._mat.copy()
        self._inverse = None

    @property
    def inverse(self):
        """Return the inverse, compute and cache if not set"""
        if self._inverse is None:
            self._inverse = np.linalg.inv(self._mat.ndmat)
        return self._inverse

    @classmethod
    def identity(cls):
        return Transform()

    def reset(self):
        self._mat = Matrix()
        self._inverse = None
        return self

    @staticmethod
    def translate_mat(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @staticmethod
    def scale_mat(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    @staticmethod
    def rotate_mat(theta):
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        return np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])

    def translate(self, dx, dy):
        self._mat.pre_dot_eq(self.translate_mat(dx, dy))
        self._inverse = None
        return self

    def scale(self, sx, sy):
        self._mat.pre_dot_eq(self.scale_mat(sx, sy))
        self._inverse = None
        return self

    def scale_vals(self) -> Tuple[float, float]:
        """Return scale x, y based on lengths of x-column and y-column"""
        m = self._mat.ndmat
        x, y = np.hypot(m[0], m[1])[0:2]
        return float(x), float(y)

    def scale_bbox(self, start: BBox, target: BBox, keep_aspect=False):
        """Update matrix to transform start BBox into target BBox (scale+translate)"""
        if keep_aspect:
            # Retain aspect ratio and center transform
            scale = min(target.width / start.width, target.height / start.height)
            self.translate(-start.xl, -start.yl)
            self.scale(scale, scale)
            w_extra = target.width - start.width*scale
            h_extra = target.height - start.height*scale
            self.translate(target.xl + w_extra/2, target.yl + h_extra/2)
        else:
            self.translate(-start.xl, -start.yl)
            self.scale(target.width / start.width, target.height / start.height)
            self.translate(target.xl, target.yl)
        return self

    def rotate(self, degrees):
        """Counterclockwise in degrees"""
        self._mat.pre_dot_eq(self.rotate_mat(math.radians(degrees)))
        self._inverse = None
        return self

    def rotate_about(self, angle, center: BasePoint):
        """Update matrix to rotate about center (instead of 0, 0)"""
        # return self.compose(Transform().translate(*-center).rotate(angle).translate(*center))
        xform = Transform().translate(*-center).rotate(angle).translate(*center)
        self._mat.pre_dot_eq(xform._mat.ndmat)
        return self

    def compose(self, transmat: Union['Transform', Matrix]):
        """Compose (dot-product) this transform with another"""
        transmat = transmat._mat if isinstance(transmat, Transform) else transmat
        self._mat.post_dot_eq(transmat.ndmat)
        self._inverse = None
        return self

    def compose_outer(self, transmat: Union['Transform', Matrix]):
        """Compose (dot-product) this transform with another"""
        transmat = transmat._mat if isinstance(transmat, Transform) else transmat
        self._mat.pre_dot_eq(transmat.ndmat)
        self._inverse = None
        return self

    def is_simple(self):
        """True if this xform represents only translation, scale, and 90-degree rotations"""
        ndmat = self._mat.ndmat

        return near_zero(ndmat[0][1]) and near_zero(ndmat[1][0]) or (near_zero(ndmat[0][0]) and near_zero(ndmat[1][1]))

    def simplify(self) -> Tuple['Transform', 'Transform']:
        """Decompose this transformation into translation+90-degree rotation and scale/skew/rotate matrix"""
        a, b, c, d, e, f = self.as_6tuple()
        a = 0 if near_zero(a) else a
        b = 0 if near_zero(b) else b
        d = 0 if near_zero(d) else d
        e = 0 if near_zero(e) else e
        if self.is_simple():
            return Transform((a, b, c, d, e, f)), Transform()
        else:
            return Transform((1, 0, c, 0, 1, f)), Transform((a, b, 0, d, e, 0))

    @staticmethod
    def canvas_fit(canvas_size, zoom=1.0, zero_zero=(0, 0)):
        """Create a Transform() to map (right, up) modeling space to (right, down) screen space"""
        return Transform().scale(zoom, zoom).translate(*zero_zero).scale(1, -1).translate(0, canvas_size[1])

    def transform_np_pts(self, np_pts: NumpyArray) -> NumpyArray:
        """Transform numpy points to numpy points"""
        return np.dot(self._mat.ndmat, np_pts).T

    def transform_width(self, width):
        """Scale width by average of x & y scales"""
        sx, sy = self.scale_vals()
        return width * (abs(sx) + abs(sy)) / 2

    def untransform(self, x, y):
        p = np.array([x, y, 1.]).T
        # print('transform: p=', p)
        pp = np.dot(self.inverse, p).T
        return pp[0], pp[1]

    def untransform_pt(self, pt):
        return Point(*self.untransform(*pt))

    def untransform_pts(self, pts: List[Tuple[Number, Number]]):
        """Transform a list"""
        pts = np.array([(x, y, 1) for x, y in pts]).T
        # print('transform: pts=', pts)
        pts = np.dot(self.inverse, pts).T
        return [(x, y) for x, y, z in pts]

    def push(self):
        """
            Push/pop can be used stand-alone, or push() can be used in a 'with' statement::
                with m.push().translate(100,100).rotate(45):
                    m.transform(...)

                m.push()
                m.translate(100, 100).rotate(45)
                m.transform(...)
                m.pop()
        :return:	self
        """
        self.stack.append(self._mat.copy())
        return self

    def pop(self):
        if not self.stack:
            raise EmptyStackError()
        self._mat = self.stack.pop()
        self._inverse = None

    def __enter__(self):
        """Nothing to do on enter, as the assumption is that push() was recently called"""
        if not self.stack:
            raise EmptyStackError("__enter__ called before .push() called")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pop()


if __name__ == '__main__':      # pragma: no cover
    pass
    # test()
