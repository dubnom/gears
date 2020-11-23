"""
Basic geometry objects
"""

import math
import typing
from abc import ABC, abstractmethod
from itertools import islice, cycle as iter_cycle
from typing import overload, Iterable, Collection, Union
from PIL import ImagePath

from .typing import *


Points = Iterable[Union['BasePoint', Number, Tuple[Number, Number]]]
PointList = Iterable['BasePoint']
ImagePathType = type(ImagePath.Path([]))


__all__ = [
    'BasePoint',
    'Point', 'Points', 'PointList', 'PointCalc', 'PointRelative',
    'Vector', 'VectorRelative',
    'BBox', 'Line',
    'ImagePathType',
    'xyiter', 'flatten_xy', 'polygon_area', 'iter_rotate',
]


def xyiter(iterable):
    it = iter(iterable)
    for x in it:
        yield x, next(it)


def flatten_xy(iterable):
    """Flatten [(x, y)...] to [x, y, ...]"""
    return [v for xy in iterable for v in xy]


def iter_rotate(data: Union[str, Collection, Iterable], rotations=2, offset=0, cycle=True) -> Iterable:
    """
        Return zip(list[offset:], list[offset+1:]...).  Usage:
            for x, y, z in iter_rotate(data, 3, -1):
                ...
        :param data:    list or string
        :param rotations:   number of times to repeat the data
        :param offset:      offset for first rotation
        :param cycle:       cycle back to start
        :return: zip()
    """
    offset = offset % len(data)
    if cycle:
        return zip(*(islice(iter_cycle(data), idx+offset, idx+offset+len(data)) for idx in range(rotations)))
    else:
        return zip(*(islice(iter_cycle(data), idx+offset, idx+offset+len(data)-1) for idx in range(rotations)))


class SupportsRound(typing.SupportsRound):
    """
        PyCharm has old version of typing.pyi with -> int return.
        Using this hacky class somehow fixes the issue.  Or, just edit typing.pyi to
        comment out the offending method:

            @runtime_checkable
            class SupportsRound(Protocol[_T_co]):
                # @overload
                # @abstractmethod
                # def __round__(self) -> int: ...
                @overload
                @abstractmethod
                def __round__(self, ndigits: int) -> _T_co: ...
    """
    def __round__(self, n=None):    # pragma: no cover
        return 0


class Vector(typing.SupportsRound):
    """
        The difference of two points.
        Operations supported:
            P-P -> V
            V+-P -> P
            P+-V -> P
            V+-V -> V
            V*/N -> V
        Not allowed: P+P, P*N
    """

    def __init__(self, x=0.0, y=0.0):
        self._x = x
        self._y = y

    def __str__(self):
        return 'V%s' % str(self.xy())

    def __repr__(self):
        x, y = self.xy()
        return '%s(%s, %s)' % (self.__class__.__name__, x, y)

    def __bool__(self):
        """Return True if x or y is non-zero"""
        return bool(self._x or self._y)

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return type(self) is type(other) and self._x == other._x and self._y == other._y

    def close(self, other, max_delta=1e-11):
        return (self - other).length() < max_delta

    def copy(self) -> 'Vector':
        return Vector(self._x, self._y)

    def fixed(self) -> 'Vector':
        return Vector(*self.xy())

    def __round__(self, n: int = 0):
        return Vector(round(self._x, n), round(self._y, n))

    def round(self, digits=0) -> 'Vector':
        return round(self, digits)

    def xy(self) -> tuple:
        return self._x, self._y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def set(self, v: 'Vector') -> None:
        self._x, self._y = v.xy()

    def setxy(self, x, y) -> None:
        self._x = x
        self._y = y

    def length(self) -> float:
        return math.hypot(*self.xy())

    def angle(self):
        return math.degrees(math.atan2(self.y, self.x))

    def unit(self) -> 'Vector':
        """Return the unit version of this vector"""
        length = self.length()
        if length == 0:
            raise ZeroDivisionError('Cannot compute Vector(0,0).unit()')
        x, y = self.xy()
        return Vector(x / length, y / length)

    def normal(self) -> 'Vector':
        """Return the normal vector.  I.e., 90 degrees CW or cross-product with Z unit vector"""
        x, y = self.xy()
        return Vector(y, -x)

    def dot(self, other: 'Vector') -> float:
        """Dot product of this Vector with another Vector"""
        sx, sy = self.xy()
        ox, oy = other.xy()
        return sx*ox + sy*oy

    def cross(self, other: 'Vector') -> float:
        """2-d cross product (scalar value)"""
        sx, sy = self.xy()
        ox, oy = other.xy()
        return sx*oy - sy*ox

    def __iter__(self):
        return iter(self.xy())

    @overload
    def __add__(self, vector: 'Vector') -> 'Vector': ...

    @overload
    def __add__(self, point: 'BasePoint') -> 'BasePoint': ...

    def __add__(self, other):
        if hasattr(other, 'add_vector'):
            return other.add_vector(self)
        if isinstance(other, Vector):
            x, y = self.xy()
            ox, oy = other.xy()
            return Vector(x + ox, y + oy)
        return NotImplemented

    def __sub__(self, other):
        return other.sub_vector(self)

    def sub_vector(self, other) -> 'Vector':
        """returns other-self, since this is called from other.sub(this)"""
        x, y = self.xy()
        ox, oy = other.xy()
        return Vector(ox - x, oy - y)

    def __mul__(self, other) -> 'Vector':
        x, y = self.xy()
        return Vector(x * other, y * other)

    def __rmul__(self, other) -> 'Vector':
        x, y = self.xy()
        return Vector(x * other, y * other)

    def __truediv__(self, other) -> 'Vector':
        x, y = self.xy()
        return Vector(x / other, y / other)

    def __neg__(self) -> 'Vector':
        x, y = self.xy()
        return Vector(-x, -y)


class VectorRelative(Vector):
    """
        The calculated difference of two points.
    """

    def __init__(self, p1: 'BasePoint', p2: 'BasePoint'):
        super().__init__(0, 0)   # yuck
        self._p1 = p1
        self._p2 = p2

    def __str__(self):
        return 'V(%s-%s)' % (self._p1, self._p2)

    def __repr__(self):
        return '%s(%s, %s)' % (self.__class__.__name__, self._p1, self._p2)

    def __bool__(self):
        """Return True if x or y is non-zero"""
        x, y = self.xy()
        return bool(x or y)

    def copy(self) -> 'VectorRelative':
        return VectorRelative(self._p1, self._p2)

    def __round__(self, n: int = 0):
        return VectorRelative(round(self._p1, n), round(self._p2, n))

    def round(self, digits=0) -> 'Vector':
        return round(self, digits)

    def xy(self) -> tuple:
        return (self._p1 - self._p2).xy()

    @property
    def x(self):
        return self.xy()[0]

    @property
    def y(self):
        return self.xy()[1]

    def __iter__(self):
        return iter(self.xy())

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return (
            self.__class__ is other.__class__ and
            self._x == other._x and self._y == other._y and
            self._p1 == other._p1 and self._p2 == other._p2
        )


class BasePoint(ABC, SupportsRound):
    """Base class for point classes"""

    def __init__(self, name=None):
        self.name = name

    def __str__(self):
        return str(self.xy())

    def _repr(self, extra=''):
        x, y = self.xy()
        return '%s(%s, %s%s%s)' % (self.__class__.__name__, x, y, extra, self._namestr())

    def _namestr(self):
        return ', %r' % self.name if self.name else ''

    def __repr__(self):
        return self._repr()

    @abstractmethod
    def __eq__(self, other):
        return type(self) is type(other) and self.name == other.name

    def close(self, other: 'BasePoint', max_delta=1e-11):
        return (self - other).length() < max_delta

    @abstractmethod
    def copy(self) -> 'BasePoint':                  # pragma: no cover
        ...

    def __round__(self, n=None):
        return round(self.fixed(), n)

    def round(self, digits=0) -> 'BasePoint':
        return round(self, digits)

    def add_vector(self, other):
        x, y = self.xy()
        ox, oy = other.xy()
        return Point(x + ox, y + oy)

    def sub_vector(self, other):
        unused(self, other)
        # Implementation of other-self where other is a Vector
        return NotImplemented       # Vector - Point is not allowed

    def fixed(self) -> 'Point':
        return Point(*self.xy(), 'fixed_' + self.name if self.name else None)

    def xy(self):
        return 0, 0

    def mid(self, other):
        """Midpoint between self and other"""
        sx, sy = self.xy()
        ox, oy = other.xy()
        return Point((sx+ox)/2, (sy+oy)/2)

    @property
    def x(self):
        return self.xy()[0]

    @property
    def y(self):
        return self.xy()[1]

    def __iter__(self):
        return iter(self.xy())

    @staticmethod
    def parse(pts: Points, allow_thruples=False):
        """
            Parse sequence of Point objects, xy-tuples, or xy separate values
            :returns: List of tuples or points.  Note: tuples might be thruples if allow is True.
        """
        out = []
        it = iter(pts)
        for a in it:
            if isinstance(a, Number):
                try:
                    y = next(it)
                except StopIteration:
                    raise ValueError('Missing y value for x=%r' % a)
                if not isinstance(y, Number):
                    raise ValueError('y value for x=%s must be a number' % repr(a))
                val = (a, y)
            elif isinstance(a, BasePoint):
                val = a
            elif isinstance(a, tuple):
                if allow_thruples:
                    if not len(a) in [2, 3]:
                        raise ValueError('tuples must be x,y or x,y,a, not %s' % repr(a))
                elif len(a) != 2:
                    raise ValueError('tuples must be x,y, not %s' % repr(a))
                val = a
            else:
                raise ValueError('Expected x,y or (x,y) or Point, not %s' % repr(a))
            out.append(val)
        return out

    @overload
    def __add__(self, other: Vector) -> 'Point': ...

    def __add__(self, other):
        if isinstance(other, Vector):
            return other + self

        return NotImplemented
        # raise TypeError("Cannot add %s and %s" % (self.__class__.__name__, other.__class__.__name__))

    @overload
    def __sub__(self, point: 'BasePoint') -> Vector: ...

    @overload
    def __sub__(self, vector: Vector) -> 'BasePoint': ...

    def __sub__(self, other) -> Union['BasePoint', Vector]:
        x, y = self.xy()
        ox, oy = other.xy()
        kind = Point if isinstance(other, Vector) else Vector
        return kind(x - ox, y - oy)

    def __neg__(self) -> 'Point':
        x, y = self.xy()
        return Point(-x, -y)


def polygon_area(points: PointList):
    """Compute the area of polygon represented by points.  >0 means CCW, <0 means CW"""
    return sum(a.x*b.y - a.y*b.x for a, b in iter_rotate(points))/2


class Point(BasePoint):
    """
        Simple point class and base class for other point classes
        Point types:
            * Absolute - (x,y) represent the actual location of the point
            * Anchor - an Absolute point that should not be moved by simple animations
            * Relative - (x,y) represent a delta from the location of <base>.  I.e., self.xy() is base.xy+self._xy
            * Calc - computed (x,y) of this point is scale*base.dxy+base.base.xy
    """

    def __init__(self, x, y, name=None):
        super().__init__(name)
        self._x = x
        self._y = y

    def __eq__(self, other):
        # noinspection PyProtectedMember
        return super().__eq__(other) and self._x == other._x and self._y == other._y

    def __round__(self, n=None):
        return Point(round(self._x, n), round(self._y, n), self.name)

    def round(self, digits=0):
        return Point(round(self._x, digits), round(self._y, digits), self.name)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def xy(self):
        """Return actual (x,y) values for this point"""
        return self._x, self._y

    def copy(self) -> 'Point':
        """Return a copy of this point"""
        return Point(self._x, self._y)

    def restore(self, other: BasePoint):
        self._x, self._y = other.xy()

    def set(self, p: BasePoint):
        """Set this point from other point"""
        self._x, self._y = p.xy()

    def setxy(self, x, y):
        """Set this point from x & y"""
        self._x = x
        self._y = y


class PointRelative(BasePoint):
    def __init__(self, dx, dy, base: BasePoint, name=None):
        super().__init__(name)
        self.dx = dx
        self.dy = dy
        self.base = base

    def __repr__(self):
        return 'PointRelative(%s, %s, %r%s)' % (self.dx, self.dy, self.base, self._namestr())

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self.dx == other.dx and self.dy == other.dy and
            self.base == other.base
        )

    def copy(self) -> 'PointRelative':
        return PointRelative(self.dx, self.dy, self.base, name=self.name)

    @staticmethod
    def fromxy(x, y, base):
        basex, basey = base.xy()
        return PointRelative(x - basex, y - basey, base)

    def xy(self):
        basex, basey = self.base.xy()
        return self.dx + basex, self.dy + basey


class PointCalc(BasePoint):
    def __init__(self, base: BasePoint, scale, name: str = None):
        super().__init__(name)
        assert hasattr(base, 'base')
        self.base = cast(PointRelative, base)
        self.scale = scale

    def __repr__(self):
        if self.name:
            return 'PointCalc(%r, %s, %r)' % (self.base, self.scale, self.name)
        else:
            return 'PointCalc(%r, %s)' % (self.base, self.scale)

    def __eq__(self, other):
        return (
            super().__eq__(other) and
            self.scale == other.scale and
            self.base == other.base
        )

    def copy(self) -> 'PointCalc':
        return PointCalc(self.base, self.scale, name=self.name)

    def xy(self):
        basex, basey = self.base.base.xy()
        return self.scale * self.base.dx + basex, self.scale * self.base.dy + basey


class BBox(object):
    """
    A bounding box.

    States:
        * is_none: BBox has no coordinates: BBox(None)
        * is_empty: BBox has zero area, but might have coordinates, e.g. BBox(1, 1, 2, 1)
        * otherwise: BBox has non-zero area

    Algebra operations:
        * Contains: True if this BBox contains other Point/BBox
            - implies that all(.contains(other.p{ll,lh,hl,hh})) is True
            - implies that .intersection(other)==other
        * Touches: True if this BBox touches other Point/BBox, but does not contain it
            - implies that .intersection(other).area == 0
            - implies that .overlaps(other) is False
            - implies that any(.touches(other.p{ll,lh,hl,hh})) is True
        * Overlaps: True if this BBox overlaps other BBox
            - implies that .intersection(other) is True
            - implies that if .overlap(other) != other, then any(.contains(other.p{ll,lh,hl,hh})) is True
        * Intersection: BBox of intersection with other
    .is_none operations:
        * Contains/Touches/Overlaps is always False
        * Intersection is always empty
    """
    __slots__ = ('xl', 'yl', 'xh', 'yh')

    @overload
    def __init__(self, empty: None):                                        # pragma: no cover
        ...

    @overload
    def __init__(self, coords: Tuple[Number, Number, Number, Number]):      # pragma: no cover
        ...

    @overload
    def __init__(self, xl: Number, yl: Number, xh: Number, yh: Number):     # pragma: no cover
        ...

    @overload
    def __init__(self, p1: BasePoint, p2: BasePoint):                       # pragma: no cover
        ...

    def __init__(self, xl, yl=None, xh=None, yh=None):
        """
            Construct a BBox:
                * BBox(None)
                * BBox((x, y, x2, y2))
                * BBox(x, y, x2, y2)
                * BBox(p1, p2)
        """
        if xl is None:
            # This is a no-coordinate BBox
            self.xl = self.yl = self.xh = self.yh = None
        else:
            usage_fmt = 'Expected BBox(tuple), BBox(x,y,x,y), or BBox(pt,pt), got: %s'
            if yl is None:
                if xh or yh or not isinstance(xl, tuple):
                    raise TypeError(usage_fmt % repr((xl, yl, xh, yh)))
                xl, yl, xh, yh = xl
            elif xh is None:
                if yh or not isinstance(xl, BasePoint) or not isinstance(yl, BasePoint):
                    raise TypeError(usage_fmt % repr((xl, yl, xh, yh)))
                xl, yl, xh, yh = xl.xy() + yl.xy()
            self.xl, self.xh = min(xl, xh), max(xl, xh)
            self.yl, self.yh = min(yl, yh), max(yl, yh)

    def __str__(self):
        return str(self.as_tuple())

    def __repr__(self):
        return 'BBox(xl=%r, yl=%r, xh=%r, yh=%r)' % self.as_tuple()

    def __eq__(self, other):
        return type(self) is type(other) and \
               self.xl == other.xl and self.yl == other.yl and \
               self.xh == other.xh and self.yh == other.yh

    def __getitem__(self, item):
        return getattr(self, self.__slots__[item])

    def __setitem__(self, item, value):
        setattr(self, self.__slots__[item], value)

    def __round__(self, n=None):
        return BBox(round(self.xl, n), round(self.yl, n), round(self.xh, n), round(self.yh, n))

    def sort(self):
        self.xl, self.xh = min(self.xl, self.xh), max(self.xl, self.xh)
        self.yl, self.yh = min(self.yl, self.yh), max(self.yl, self.yh)

    def copy(self):
        return BBox(self.xl, self.yl, self.xh, self.yh)

    def expand(self, expand_x, expand_y=None):
        """Expand/shrink a BBox."""
        expand_y = expand_x if expand_y is None else expand_y
        expand_x = max(expand_x, -self.width/2)
        expand_y = max(expand_y, -self.height/2)
        return BBox(self.xl-expand_x, self.yl-expand_y, self.xh+expand_x, self.yh+expand_y)

    @property
    def is_none(self):
        """Return True if BBox has no coordinates"""
        return self.xl is None

    @property
    def is_empty(self):
        """Return True if BBox has zero area"""
        return self.xl is None or self.xl == self.xh or self.yl == self.yh

    @property
    def p1(self):
        """Same as self.pll.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xl, self.yl)

    @property
    def p2(self):
        """Same as self.phh.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xh, self.yh)

    @property
    def pll(self):
        """Lower left==x-low, y-low.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xl, self.yl)

    @property
    def plh(self):
        """Lower right==x-low, y-high.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xl, self.yh)

    @property
    def phl(self):
        """Upper left==x-high, y-low.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xh, self.yl)

    @property
    def phh(self):
        """Upper right==x-high, y-high.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point(self.xh, self.yh)

    @property
    def center(self):
        """Center.  ValueError if .is_none"""
        if self.xl is None:
            raise ValueError("BBox is Empty")
        return Point((self.xl+self.xh)/2, (self.yl+self.yh)/2)

    def __add__(self, other):
        """BBox of self and other"""
        if not isinstance(other, BBox):
            return NotImplemented
        if other.xl is None:
            return self.copy()
        if self.xl is None:
            return other.copy()
        return BBox(
            min(self.xl, other.xl), min(self.yl, other.yl),
            max(self.xh, other.xh), max(self.yh, other.yh)
        )

    @property
    def width(self):
        """Width, 0 if .is_none"""
        if self.xl is None:
            return 0
        return self.xh - self.xl

    @property
    def height(self):
        """Height, 0 if .is_none"""
        if self.xl is None:
            return 0
        return self.yh - self.yl

    def area(self):
        """Area, 0 if .is_none"""
        if self.xl is None:
            return 0
        return (self.xh-self.xl) * (self.yh-self.yl)

    def size(self) -> tuple:
        """width, height as a tuple"""
        return self.width, self.height

    def grow(self, point: BasePoint):
        """Expand this BBox to include point"""
        x, y = point.xy()
        if self.xl is None:
            self.xl = self.xh = x
            self.yl = self.yh = y
        else:
            self.xl = min(self.xl, x)
            self.xh = max(self.xh, x)
            self.yl = min(self.yl, y)
            self.yh = max(self.yh, y)

    def transformed(self, xform) -> 'BBox':
        """Return the BBox of all 4 corners transformed"""
        ll, lh, hh, hl = xform.transform_pts([self.pll, self.plh, self.phh, self.phl])
        return BBox(ll, lh) + BBox(hh, hl)

    def contains_pt(self, point: BasePoint) -> bool:
        """Return True if point is inside this BBox"""
        if self.xl is None:
            return False
        x, y = point.xy()
        return self.xl < x < self.xh and self.yl < y < self.yh

    def contains(self, other: 'BBox') -> bool:
        """Return True if other is inside (or at edge) this BBox"""
        if self.xl is None or other.xl is None:
            raise ValueError("BBox is empty")
        return self.xl <= other.xl and other.xh <= self.xh and self.yl <= other.yl and other.yh <= self.yh

    def touches_pt(self, point: BasePoint) -> bool:
        """Return True if point is on border of this BBox"""
        if self.xl is None:
            return False
        x, y = point.xy()
        return ((x == self.xl or x == self.xh) and self.yl < y < self.yh) or \
               ((y == self.yl or y == self.yh) and self.xl < x < self.xh)

    def touches_side(self, other: 'BBox') -> bool:
        """Return True if other shares a border with this BBox, but does not overlap"""
        if self.xl is None or other.xl is None:
            return False
        return ((other.xh == self.xl or other.xl == self.xh) and self.yl < other.yh and other.yl < self.yh) or \
               ((other.yh == self.yl or other.yl == self.yh) and self.xl < other.xh and other.xl < self.xh)

    def touches(self, other: 'BBox') -> bool:
        """Return True if other shares a border or corner with this BBox, but does not overlap"""
        if self.xl is None or other.xl is None:
            return False
        return ((other.xh == self.xl or other.xl == self.xh) and self.yl <= other.yh and other.yl <= self.yh) or \
               ((other.yh == self.yl or other.yl == self.yh) and self.xl <= other.xh and other.xl <= self.xh)

    def overlaps(self, other: 'BBox') -> bool:
        """Return True if this BBox has non-zero intersection with other"""
        if self.xl is None or other.xl is None:
            raise ValueError("BBox is empty")
        if self.xl >= other.xh or self.xh <= other.xl or self.yl >= other.yh or self.yh <= other.yl:
            return False
        return True

    def intersection(self, other: 'BBox') -> 'BBox':
        """Return BBox of intersection or BBox(None) if no intersection"""
        if self.xl is None or other.xl is None:
            return BBox(None)
        if self.xl > other.xh or self.xh < other.xl or self.yl > other.yh or self.yh < other.yl:
            return BBox(None)
        return BBox(max(self.xl, other.xl), max(self.yl, other.yl), min(self.xh, other.xh), min(self.yh, other.yh))

    def as_tuple(self) -> tuple:
        """Return a 4-tuple of (xl, yl, xh, yh)"""
        return self.xl, self.yl, self.xh, self.yh

    def __iter__(self):
        return iter(self.as_tuple())


class Line(object):
    """Infinite length line or line segment"""
    def __init__(self, origin: BasePoint, direction: Union[BasePoint, Vector]):
        self.origin = origin
        if isinstance(direction, Vector):
            self.direction = direction
        else:
            self.direction = direction - origin

    @property
    def p1(self):
        return self.origin

    @property
    def p2(self):
        return self.origin + self.direction

    @classmethod
    def from_pts(cls, p1: BasePoint, p2: BasePoint):
        return Line(p1, p2-p1)

    def closest(self, point: BasePoint):
        """Compute the closest point to this line"""

        to_point = point - self.origin
        d_unit = self.direction.unit()
        dot = to_point.dot(d_unit)
        return self.origin + d_unit*dot

    def intersection(self, other: 'Line'):
        """Return intersection point or None"""
        if self.direction.cross(other.direction) == 0:
            # Parallel
            raise ValueError('parallel lines')
        normal = self.direction.normal()
        t_other = normal.dot(self.origin-other.origin) / normal.dot(other.direction)
        return other.origin + other.direction*t_other

    def segment_bbox(self):
        return BBox(self.origin, self.origin+self.direction)

    def parallel(self, other):
        return self.direction.cross(other.direction) == 0

    def segment_intersection(self, other: 'Line') -> Optional[Tuple[Point, Number, Number]]:
        """Return None or (intersection point, t_self, t_other)"""
        if not self.segment_bbox().overlaps(other.segment_bbox()):
            return None
        if self.parallel(other):
            # Parallel
            # TODO-Could still intersect at point or along segment
            raise ValueError('parallel lines')
        self_normal = self.direction.normal()
        t_other = self_normal.dot(self.origin-other.origin)/self_normal.dot(other.direction)
        if t_other < 0 or t_other > 1:
            return None
        other_normal = other.direction.normal()
        t_self = other_normal.dot(other.origin-self.origin)/other_normal.dot(self.direction)
        if t_self < 0 or t_self > 1:
            return None
        debug = True
        if debug:
            p_other = other.origin+other.direction*t_other
            p_self = self.origin+self.direction*t_self
            # print('self: %s@%s  other: %s@%s  length: %s' % (
            #       p_self, t_self, p_other, t_other, (p_self-p_other).length()))
            assert (p_self-p_other).length() < 1e-7
        return other.origin + other.direction*t_other, t_self, t_other
