from math import copysign
from typing import cast

import x7.view.digi
from gear_involute import GearInvolute
from x7.geom.colors import PenBrush
from x7.geom.geom import Vector
from x7.view.digi import DigitizeController
from x7.view.modes.adddrag import ModeAddDrag
from x7.view.shapes import DigitizeShape
from view.elem_gear import ViewGear, ElemGear


class ModeAddGear(ModeAddDrag):
    """Mode for adding via a single drag operation"""

    SHAPE_NAME = 'Gear'

    def __init__(self, controller: DigitizeController):
        super().__init__(controller)
        # self.verbose = True

    def drag_begin(self, mp) -> DigitizeShape:
        """Start a drag operation at model space point mp"""
        # return ViewGear(self.controller.view, ElemEllipse('gearN', PenBrush('black'), mp, mp))
        gear = GearInvolute(teeth=3, center=mp)
        return ViewGear(self.controller.view, ElemGear('gearN', PenBrush('black'), gear))

    def drag_extend(self, start, mp, shape: 'ViewGear'):
        """Extend a drag operation to model space point mp"""
        # shape = cast(ViewGear, shape)
        elem = cast(ElemGear, shape.elem)
        gear = elem.gear
        rv: Vector = mp - gear.center
        gear.set_teeth(max(3, round(rv.length() * 2 / gear.module)))
        gear.rot = -rv.angle() / 360 * gear.teeth


x7.view.digi.test_digi()
