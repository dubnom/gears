import tkinter as tk
from PIL import ImagePath

from gear_involute import GearInvolute
from x7.geom.colors import PenBrush
from x7.geom.drawing import DrawingContext
from x7.geom.model import Elem, Path, ControlPath, DumpContext
from x7.geom.typing import *
from x7.geom.transform import *
from x7.geom.geom import *
from x7.view.details import DetailPoint, DetailFloat, DetailInt
from x7.view.digibase import DigiDraw
from x7.view.shapes import DigitizeShape
from x7.view.shapes.shape import EditHandle


# TODO-make ElemGear take a gear: GearInvolute
class ElemGear(Elem):
    def __init__(self, name: str, penbrush: PenBrush,
                 gear: GearInvolute,
                 closed=True, xform: Optional[Transform] = None):
        super().__init__(name, penbrush, closed, xform)
        self.gear = gear

    def __str__(self):
        return 'ElemGear(%r, %d, %s)' % (self.name, self.gear.teeth, self.gear.center.round(2))

    def __repr__(self):
        return 'ElemGear(%r, %d, %s)' % (self.name, self.gear.teeth, self.gear.center.round)

    def __eq__(self, other):
        return super().__eq__(other) and self.gear == other.gear

    def fixed(self):
        """Return copy with all points 'fixed'"""
        return ElemGear(self.name, self.penbrush, self.gear.copy(), self.closed, self.xform)

    def copy(self):
        """Return deep copy of self"""
        # TODO-This shouldn't really be 'fixed', but we don't have restore() for non-Point yet
        return self.fixed()

    def restore(self, other: 'ElemGear'):
        """Restore self from copy in other"""
        super().restore(other)
        self.gear.restore(other.gear)

    def transform(self, matrix: Transform):
        """Return a copy of this shape transformed by matrix.  .transform(identity) is like .copy()"""

        xform = matrix.copy().compose(self.xform)
        return ElemGear(self.name, self.penbrush, self.gear, self.closed, xform)

    def bbox_int(self):
        """Return the bounding box of this shape"""
        rv = Vector(self.gear.tip_radius, self.gear.tip_radius)
        return BBox(self.gear.center+rv, self.gear.center-rv)

    def bbox_int_update(self, bbox: BBox, context=None):
        """Update shape based on changes to bounding box, resize/scale/translate as needed"""
        raise ValueError("Can't bbox_update a curve yet")

    def points_transformed(self, matrix: Transform):
        return matrix.transform_pts(self.gear.instance().poly_at(offset=Vector(0, 0)))

    def paths(self, dc: DrawingContext) -> List[Path]:
        """
            The outside path of the gear
            TODO-Add internal circles?
        """
        with self.draw_space(dc):
            path = self.points_transformed(dc.matrix)
        path.append(path[0])
        return [Path(self.penbrush, self.closed, ImagePath.Path(path))]

    def control_paths(self, dc) -> List[ControlPath]:
        """
            Just the control points
        """
        return []

    def display(self, detail=1, prefix=''):
        """Display debug details"""
        print('%sElemGear: %r %d teeth, %.4f module' % (prefix, self.name, self.gear.teeth, self.gear.module))
        if detail:
            print('%s %s' % (prefix, self.gear))

    def dump(self, context: DumpContext) -> DumpContext:
        raise TypeError('Gear does not support dump yet')

    def as_digi_points(self):
        raise TypeError('Gear does not support as_digi_points yet')


class ViewGear(DigitizeShape):
    def __init__(self, dd: Optional[DigiDraw], gear: ElemGear):
        super().__init__(dd, gear)
        self.elem = gear        # type fix

    def details(self):
        elem = self.elem
        return super().details() + [
            None,
            DetailPoint(elem.gear, 'center'),
            DetailInt(elem.gear, 'teeth'),
            DetailFloat(elem.gear, 'module'),
            DetailFloat(elem.gear, 'relief_factor'),
            DetailFloat(elem.gear, 'pressure_angle_degrees'),
            DetailFloat(elem.gear, 'rot'),
        ]

    def menu_child_gear(self):
        from .main import ModeAddGear
        mode = ModeAddGear(self.dd.dc, parent=self)
        self.dd.mode_set(mode)
        # self.dc.view.mode_set(mode_class(self.dc))

    def context_extra(self, menu: tk.Menu):
        """Add extra menu items to select context menu based on shape type"""
        menu.add_separator()
        menu.add_command(label='Add Child Gear', command=self.menu_child_gear)

    def edit_handle_create(self) -> List[EditHandle]:
        return super().edit_handle_create()
